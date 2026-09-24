// SPDX-License-Identifier: GPL-3.0-only
//! Voxtral, Mistral's speech-understanding model, ported to Burn.
//!
//! The architecture is three pieces, each a module here:
//!
//! - [`encoder`]: the audio tower, a Whisper-style encoder — two convolutions
//!   over a 128-bin log-mel spectrogram, learnt positions, 32 pre-norm
//!   transformer layers — and the projector that folds four encoder frames into
//!   one text-sized embedding;
//! - [`transformer`]: the text decoder, a Llama/Mistral stack (RMS norm, rotary
//!   positions, grouped-query attention, SwiGLU);
//! - [`model`]: the two joined — the projected audio replaces the `[AUDIO]`
//!   placeholders of the prompt — and greedy decoding over the result.
//!
//! [`audio`] turns 16 kHz samples into the spectrogram the encoder reads.
//!
//! # Provenance
//!
//! Ported from `candle_transformers::models::voxtral` at
//! `jorge-menjivar/candle@eef47d5`, which this backend ran on before, and
//! checked against it layer by layer (see `parity`, and `parity/` at the root
//! of the repository for the candle side). The decoder stack and the
//! checkpoint adapter below
//! are adapted from the `qwen3-tts` example of `jorge-menjivar/burn` at
//! `1e9de733d` — the revision `Cargo.toml` pins — with what Qwen3 has and Llama
//! does not (per-head query/key norms, layer scales, sliding windows) taken
//! out.

pub mod audio;
pub mod config;
pub mod encoder;
pub mod model;
#[cfg(test)]
mod parity;
pub mod transformer;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::nn::{LinearConfig, LinearLayout};
use burn::tensor::{DType, TensorData, bf16, f16};
use burn_store::burn_pack::Tensor as PackTensor;
use burn_store::{ApplyResult, ModuleAdapter, ModuleContext, bridge};

/// The configuration of every linear layer here: the weight keeps the
/// column-major, `[d_output, d_input]` layout of the checkpoints.
///
/// That is not only about loading without a transpose. Decoding a token runs
/// every linear layer of the decoder on a single row, and the matmul kernels
/// for that product are very sensitive to the layout of the matrix: the Burn
/// port of Qwen3-TTS measured ~55 µs with a row-major weight against ~12 µs,
/// the memory-bandwidth limit, with a column-major one on an RTX 3090.
pub(crate) fn linear_config(d_input: usize, d_output: usize) -> LinearConfig {
    LinearConfig::new(d_input, d_output).with_layout(LinearLayout::Col)
}

/// Counts the checkpoint's bytes into `read` as each tensor's are drawn,
/// which is how far a load has got. First in the chain, so it counts what the
/// file holds rather than what a cast turns it into.
#[derive(Debug, Clone)]
pub(crate) struct ReadCounter(pub Arc<AtomicU64>);

impl ModuleAdapter for ReadCounter {
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        let read = Arc::clone(&self.0);
        let bytes = tensor.byte_len() as u64;
        let (name, dtype, shape) = (tensor.name.clone(), tensor.dtype, tensor.shape.clone());
        bridge::map_data(tensor, name, dtype, shape, move |data| {
            read.fetch_add(bytes, Ordering::Relaxed);
            data
        })
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Casts the checkpoint's bf16 weights to f16 on every core, when f16 is
/// what the model runs in.
///
/// burn-store's `FloatCastAdapter` converts one element at a time on one
/// thread, which added 8 s to every load of Voxtral Mini's 4.7 billion
/// weights. The arithmetic is the same — through f32, rounding to nearest
/// even — so the weights come out bit for bit as they did. Any other tensor
/// is left to the `FloatCastAdapter` after it.
#[derive(Debug, Clone)]
pub(crate) struct HalfCast {
    pub target: DType,
}

impl ModuleAdapter for HalfCast {
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        if self.target != DType::F16 || tensor.dtype != DType::BF16 {
            return tensor;
        }
        let (name, shape) = (tensor.name.clone(), tensor.shape.clone());
        bridge::map_data(tensor, name, DType::F16, shape, |data| bf16_to_f16(&data))
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Below this many elements a tensor is converted on the calling thread.
const PARALLEL_CAST_MIN: usize = 1 << 16;

fn bf16_to_f16(data: &TensorData) -> TensorData {
    let source = data.as_slice::<bf16>().expect("a bf16 tensor holds bf16");
    let mut out = vec![f16::ZERO; source.len()];
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(source.len().div_ceil(PARALLEL_CAST_MIN))
        .max(1);
    let chunk = source.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (from, to) in source.chunks(chunk).zip(out.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (value, cast) in from.iter().zip(to) {
                    *cast = f16::from_f32(value.to_f32());
                }
            });
        }
    });
    TensorData::new(out, data.shape.clone())
}

/// Loads the PyTorch checkpoints into modules built with [`linear_config`].
///
/// Renames the parameters of the normalization layers — `weight` and `bias` in
/// PyTorch, `gamma` and `beta` in Burn — and leaves the linear weights alone: a
/// column-major linear layer stores its weight as `[d_output, d_input]`, which
/// is exactly the PyTorch layout, so there is nothing to transpose.
#[derive(Debug, Clone, Default)]
pub(crate) struct CheckpointAdapter;

impl CheckpointAdapter {
    fn is_normalization_layer(module_type: &str) -> bool {
        matches!(
            module_type,
            "Struct:BatchNorm" | "Struct:LayerNorm" | "Struct:GroupNorm" | "Struct:RmsNorm"
        )
    }
}

impl ModuleAdapter for CheckpointAdapter {
    fn adapt(&self, mut tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        let Some(module_type) = ctx.module_type() else {
            return tensor;
        };
        if !Self::is_normalization_layer(module_type) {
            return tensor;
        }
        let start = tensor.name.rfind('.').map_or(0, |dot| dot + 1);
        let renamed = match &tensor.name[start..] {
            "weight" => "gamma",
            "bias" => "beta",
            _ => return tensor,
        };
        tensor.name.truncate(start);
        tensor.name.push_str(renamed);
        tensor
    }

    /// The store looks the parameters up under their Burn names, which the
    /// checkpoint does not use for the normalization layers.
    fn get_alternative_param_name(&self, param_name: &str, module_type: &str) -> Option<String> {
        if !Self::is_normalization_layer(module_type) {
            return None;
        }
        match param_name {
            "gamma" => Some("weight".to_string()),
            "beta" => Some("bias".to_string()),
            _ => None,
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Folds the outcome of loading every shard of a checkpoint into one verdict.
///
/// A sharded checkpoint is loaded one file at a time, each allowed to be
/// partial, so a parameter one shard does not carry is reported missing by
/// that shard even when another fills it. What is really missing is what
/// *every* shard reported missing. Tensors no parameter claims are an error
/// too: a checkpoint carrying weights this port never reads is one it does not
/// implement, and running it anyway would be quietly wrong.
pub(crate) fn check_shards(results: &[ApplyResult]) -> Result<(), String> {
    let errors: Vec<String> = results
        .iter()
        .flat_map(|r| r.errors.iter().map(ToString::to_string))
        .collect();
    if !errors.is_empty() {
        return Err(errors.join(", "));
    }
    let Some((first, rest)) = results.split_first() else {
        return Err("the checkpoint has no shards".to_string());
    };
    let missing: Vec<&str> = first
        .missing
        .iter()
        .map(|(path, _)| path.as_str())
        .filter(|path| {
            rest.iter()
                .all(|r| r.missing.iter().any(|(p, _)| p == path))
        })
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{} parameters were not found in any shard, e.g. {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        ));
    }
    let unused: Vec<&str> = results
        .iter()
        .flat_map(|r| r.unused.iter().map(String::as_str))
        .collect();
    if !unused.is_empty() {
        return Err(format!(
            "{} tensors of the checkpoint belong to no parameter, e.g. {:?}",
            unused.len(),
            &unused[..unused.len().min(5)]
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parallel_cast_matches_burns_bit_for_bit() {
        // Every bf16 bit pattern, several times over so the cast is split
        // across threads: normals, subnormals, values past f16's range, both
        // infinities and NaNs.
        let values: Vec<bf16> = (0..4 * 65_536_u32)
            .map(|bits| bf16::from_bits(u16::try_from(bits % 65_536).unwrap()))
            .collect();
        let data = TensorData::new(values, [4, 65_536]);
        let ours = bf16_to_f16(&data);
        let burns = data.convert_dtype(DType::F16);
        assert_eq!(ours.shape, burns.shape);
        let ours = ours.as_slice::<f16>().unwrap();
        let burns = burns.as_slice::<f16>().unwrap();
        for (i, (a, b)) in ours.iter().zip(burns).enumerate() {
            let same = a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
            assert!(same, "element {i}: {a} against {b}");
        }
    }
}
