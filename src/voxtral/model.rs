// SPDX-License-Identifier: GPL-3.0-only
//! The whole model: audio tower and projector, spliced into the decoder's
//! prompt, and greedy decoding over the result.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use burn::nn::{Embedding, EmbeddingConfig, Linear};
use burn::prelude::*;
use burn::tensor::DType;
use burn_store::{FloatCastAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot, SafetensorsStore};

use crate::voxtral::audio::{CHUNK_FRAMES, Features};
use crate::voxtral::config::{TextConfig, VoxtralConfig};
use crate::voxtral::encoder::{AudioEncoder, Projector};
use crate::voxtral::transformer::{Transformer, TransformerState};
use crate::voxtral::{CheckpointAdapter, ReadCounter, check_shards, linear_config};

/// End-of-sequence tokens, the set candle's `generate` stopped on.
const EOS_TOKENS: [u32; 4] = [2, 128_001, 128_009, 128_256];

/// The key/value caches are sized in steps of this many positions, so that
/// prompts of nearby lengths share one set of decoding kernels.
const CACHE_GRANULE: usize = 512;

/// Records intermediate outputs by name, for the layer-by-layer comparison
/// with candle. Off everywhere but in that test, where it costs a device
/// read per tap; off, a tap is a branch.
#[derive(Debug, Default)]
pub struct Taps {
    records: Option<Vec<(String, Vec<f32>)>>,
}

impl Taps {
    /// Taps that record nothing.
    pub fn off() -> Self {
        Self::default()
    }

    /// Taps that record everything, in f32.
    #[cfg(test)]
    pub fn on() -> Self {
        Self {
            records: Some(Vec::new()),
        }
    }

    pub fn record<const D: usize>(&mut self, name: &str, tensor: &Tensor<D>) {
        self.record_with(|| name.to_string(), tensor);
    }

    /// [`Self::record`] with a name only built when something is recorded.
    pub fn record_with<const D: usize>(
        &mut self,
        name: impl FnOnce() -> String,
        tensor: &Tensor<D>,
    ) {
        if let Some(records) = &mut self.records {
            let values = tensor
                .clone()
                .cast(DType::F32)
                .into_data()
                .try_to_vec::<f32>()
                .expect("an f32 read");
            records.push((name(), values));
        }
    }

    #[cfg(test)]
    pub fn into_records(self) -> Vec<(String, Vec<f32>)> {
        self.records.unwrap_or_default()
    }
}

#[derive(Module, Debug)]
struct TextBackbone {
    embed_tokens: Embedding,
    transformer: Transformer,
}

#[derive(Module, Debug)]
struct LanguageModel {
    model: TextBackbone,
    /// `None` when the head reuses the token embeddings.
    lm_head: Option<Linear>,
}

impl LanguageModel {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            model: TextBackbone {
                embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
                transformer: Transformer::init(cfg, device),
            },
            lm_head: (!cfg.tie_word_embeddings).then(|| {
                linear_config(cfg.hidden_size, cfg.vocab_size)
                    .with_bias(false)
                    .init(device)
            }),
        }
    }

    fn embed(&self, ids: &[u32], device: &Device) -> Tensor<3> {
        let ids: Vec<i64> = ids.iter().copied().map(i64::from).collect();
        let len = ids.len();
        let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), device);
        self.model.embed_tokens.forward(ids)
    }

    /// The logits of `hidden` `(1, 1, hidden)`, in f32: `(vocab)`.
    fn logits(&self, hidden: Tensor<3>) -> Tensor<1> {
        let logits = match &self.lm_head {
            Some(head) => head.forward(hidden),
            None => hidden.matmul(
                self.model
                    .embed_tokens
                    .weight
                    .val()
                    .transpose()
                    .unsqueeze::<3>(),
            ),
        };
        let vocab = logits.dims()[2];
        logits.cast(DType::F32).reshape([vocab])
    }
}

// The field names are the checkpoint's top-level keys, which is how the store
// finds their weights.
#[allow(clippy::struct_field_names)]
#[derive(Module, Debug)]
struct Model {
    audio_tower: AudioEncoder,
    multi_modal_projector: Projector,
    language_model: LanguageModel,
}

/// Maps the checkpoint's names onto the modules': the decoder layers and the
/// final norm live one level down, in [`Transformer`].
fn remapper() -> Result<KeyRemapper, String> {
    KeyRemapper::from_patterns(vec![(
        r"^language_model\.model\.(layers|norm)\.".to_string(),
        "language_model.model.transformer.$1.".to_string(),
    )])
    .map_err(|err| format!("invalid remapping: {err}"))
}

/// A loaded Voxtral checkpoint.
#[derive(Debug)]
pub struct Voxtral {
    model: Model,
    config: VoxtralConfig,
    device: Device,
    dtype: DType,
}

impl Voxtral {
    /// Build the model and fill it from the checkpoint's shards, casting every
    /// weight to `dtype`. `read` counts the checkpoint's bytes as they are
    /// read.
    pub fn load(
        config: &VoxtralConfig,
        shards: &[PathBuf],
        dtype: DType,
        device: &Device,
        read: &Arc<AtomicU64>,
    ) -> Result<Self, String> {
        // The weights go to the persistent pool: exact-fit slices that live
        // for the process. Left to the dynamic pools they would share pages
        // with the activations, and those pools keep whatever the worst
        // moment of a workload asked for.
        let model = device.memory_persistent_allocations((), |()| {
            Self::load_weights(config, shards, dtype, device, read)
        })?;
        Ok(Self {
            model,
            config: config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    fn load_weights(
        config: &VoxtralConfig,
        shards: &[PathBuf],
        dtype: DType,
        device: &Device,
        read: &Arc<AtomicU64>,
    ) -> Result<Model, String> {
        // Parameters are initialized lazily, so nothing is allocated for the
        // random weights the checkpoint then replaces.
        let mut model = Model {
            audio_tower: AudioEncoder::init(&config.audio_config, device),
            multi_modal_projector: Projector::init(config, device),
            language_model: LanguageModel::init(&config.text_config, device),
        };
        let mut results = Vec::with_capacity(shards.len());
        for shard in shards {
            let mut store = SafetensorsStore::from_file(shard)
                .with_from_adapter(
                    ReadCounter(Arc::clone(read))
                        .chain(CheckpointAdapter)
                        .chain(FloatCastAdapter::to(dtype)),
                )
                .remap(remapper()?)
                .allow_partial(true);
            results.push(
                model
                    .load_from(&mut store)
                    .map_err(|err| format!("loading {}: {err}", shard.display()))?,
            );
        }
        check_shards(&results)?;
        Ok(model)
    }

    /// Hand the device memory nothing holds any more back to the driver.
    ///
    /// The pools otherwise keep the most any run has asked of them: the
    /// activations of the largest transcription so far, and on the first run
    /// of a shape the buffers of every candidate kernel its autotune
    /// benchmarked — several gigabytes over what a transcription needs.
    ///
    /// Synced after. On CUDA the pages are freed in stream order, into the
    /// driver's own pool, which hands memory back to the system at the next
    /// synchronization: without it, an idle backend still holds it.
    pub fn release_scratch(&self) {
        self.device.memory_cleanup();
        if let Err(err) = self.device.sync() {
            log::warn!("syncing the device after releasing its memory failed: {err}");
        }
    }

    pub fn config(&self) -> &VoxtralConfig {
        &self.config
    }

    /// The audio embeddings of `features`: `(chunks * tokens_per_chunk,
    /// text_hidden)`, in the order the prompt's placeholders take them.
    pub fn encode_audio(&self, features: &Features, taps: &mut Taps) -> Tensor<2> {
        let mels = self.config.audio_config.num_mel_bins;
        let data = TensorData::new(features.data.clone(), [features.chunks, mels, CHUNK_FRAMES]);
        let features = Tensor::<3>::from_data(data, &self.device).cast(self.dtype);
        taps.record("mel", &features);
        let encoded = self.model.audio_tower.forward(features, taps);
        let audio = self.model.multi_modal_projector.forward(encoded);
        taps.record("proj", &audio);
        audio
    }

    /// The prompt's embeddings with `audio` in place of its placeholders,
    /// which must be one run exactly as long as `audio`.
    fn splice(&self, input_ids: &[u32], audio: Tensor<2>) -> Result<Tensor<3>, String> {
        let placeholder = self.config.audio_token_id;
        let start = input_ids
            .iter()
            .position(|&t| t == placeholder)
            .ok_or("the prompt has no audio placeholder")?;
        let end = start
            + input_ids[start..]
                .iter()
                .take_while(|&&t| t == placeholder)
                .count();
        let [rows, _] = audio.dims();
        if end - start != rows || input_ids[end..].contains(&placeholder) {
            return Err(format!(
                "the prompt's audio placeholders do not form one run of the {rows} the audio fills"
            ));
        }
        let lm = &self.model.language_model;
        Ok(Tensor::cat(
            vec![
                lm.embed(&input_ids[..start], &self.device),
                audio.unsqueeze::<3>(),
                lm.embed(&input_ids[end..], &self.device),
            ],
            1,
        ))
    }

    /// Greedy decoding of `input_ids` with `audio` spliced in. Returns the
    /// generated tokens, the end-of-sequence token included when it came.
    ///
    /// `forced` feeds those tokens back instead of the model's own, and stops
    /// after them: the parity test uses it to compare every step's logits with
    /// candle's even where a near-tie picks differently, and the warm-up to
    /// run decoding steps whatever the model makes of silence. Either way the
    /// caches are sized by `max_new_tokens`, so a forced run compiles exactly
    /// the kernels a real one uses.
    pub fn generate(
        &self,
        input_ids: &[u32],
        audio: Tensor<2>,
        max_new_tokens: usize,
        forced: Option<&[u32]>,
        taps: &mut Taps,
    ) -> Result<Vec<u32>, String> {
        let embeds = self.splice(input_ids, audio)?;
        taps.record("dec.embeds", &embeds);
        let prompt_len = input_ids.len();
        let steps = forced.map_or(max_new_tokens, <[u32]>::len);
        if steps > max_new_tokens {
            return Err(format!(
                "{steps} forced tokens are more than the {max_new_tokens} the caches are sized for"
            ));
        }
        let capacity = (prompt_len + max_new_tokens).next_multiple_of(CACHE_GRANULE);
        let mut state =
            TransformerState::new(&self.config.text_config, capacity, self.dtype, &self.device);
        let lm = &self.model.language_model;
        let hidden = lm.model.transformer.forward(embeds, 0, &mut state, taps);
        let mut logits = lm.logits(hidden.narrow(1, prompt_len - 1, 1));

        let mut tokens = Vec::new();
        for step in 0..steps {
            taps.record_with(|| format!("logits.{step}"), &logits);
            // A failed read — the device out of memory, say — is the request's
            // error, not a panic: this runs under the engine's lock, and a
            // panic would poison it for every request after.
            let token = logits
                .argmax(0)
                .try_into_scalar::<i64>()
                .map_err(|err| format!("reading the next token failed: {err}"))?;
            let token = u32::try_from(token).map_err(|_| "the argmax is not a token id")?;
            tokens.push(token);
            let next = forced.map_or(token, |f| f[step]);
            if forced.is_none() && (EOS_TOKENS.contains(&token) || tokens.ends_with(&[0; 5])) {
                break;
            }
            if step + 1 == steps {
                break;
            }
            let pos = i64::try_from(prompt_len + step).map_err(|_| "a position past i64")?;
            let pos = Tensor::<1, Int>::from_data([pos], &self.device);
            let embeds = lm.embed(&[next], &self.device);
            let hidden = lm.model.transformer.forward_at(embeds, &pos, &mut state);
            logits = lm.logits(hidden);
        }
        Ok(tokens)
    }
}
