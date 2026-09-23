// SPDX-License-Identifier: GPL-3.0-only
//! 16 kHz samples to the log-mel spectrogram the audio tower reads.
//!
//! This is the whisper.cpp algorithm candle's Voxtral used, ported operation
//! for operation — the same radix-2 FFT with a DFT under odd sizes, the same
//! order of accumulation — so that the features are bit-for-bit those the
//! candle backend computed. Two of its choices differ from the Hugging Face
//! feature extractor the model was trained with, and are kept deliberately,
//! for parity with what this backend has always sent the model:
//!
//! - frames start at their hop rather than centred on it, and the power of
//!   every bin but the first and the Nyquist one counts its mirror image too;
//! - one extra half-chunk of silence is appended before framing, so a clip of
//!   `n` whole chunks becomes `n + 1` chunks, the last half silence and half
//!   zero padding.
//!
//! Plain Rust on the host: a few milliseconds a clip, next to a model that
//! takes hundreds.

use std::f32::consts::PI;

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
/// Samples in the 30-second chunks the model reads audio in.
pub const CHUNK_SAMPLES: usize = 480_000;
/// Mel frames per chunk.
pub const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP_LENGTH;
/// The frames whisper.cpp pads the spectrogram with and rounds it up to:
/// half a chunk.
const PAD_FRAMES: usize = CHUNK_FRAMES / 2;

/// The mel filter bank, `N_MELS` rows of `N_FFT / 2 + 1` weights, as
/// little-endian f32.
const MEL_FILTERS: &[u8] = include_bytes!("../data/melfilters128.bytes");

/// Decode the embedded mel filter bank.
pub fn mel_filters() -> Vec<f32> {
    MEL_FILTERS
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Pad `audio` up to a whole multiple of `chunk` samples, zero-filling the
/// tail. Input already aligned to `chunk` (including empty input) is returned
/// as-is.
pub fn pad_to_chunk(audio: &[f32], chunk: usize) -> Vec<f32> {
    let mut padded = audio.to_vec();
    padded.resize(audio.len().div_ceil(chunk) * chunk, 0.0);
    padded
}

/// The spectrogram of a clip, cut into the chunks the encoder reads:
/// `(chunks, N_MELS, CHUNK_FRAMES)`, row-major.
#[derive(Debug, Clone)]
pub struct Features {
    pub data: Vec<f32>,
    pub chunks: usize,
}

/// The features of `samples`, which should already be padded to whole chunks
/// (see [`pad_to_chunk`]).
pub fn extract_features(samples: &[f32], filters: &[f32]) -> Features {
    let mel = log_mel_spectrogram(samples, filters);
    let frames = mel.len() / N_MELS;
    let chunks = frames.div_ceil(CHUNK_FRAMES);
    // `(N_MELS, frames)` zero-padded to whole chunks, reshaped to
    // `(N_MELS, chunks, CHUNK_FRAMES)` and transposed to put the chunks first.
    let mut data = vec![0.0; chunks * N_MELS * CHUNK_FRAMES];
    for chunk in 0..chunks {
        let start = chunk * CHUNK_FRAMES;
        let len = CHUNK_FRAMES.min(frames - start);
        for m in 0..N_MELS {
            let from = &mel[m * frames + start..m * frames + start + len];
            let to = (chunk * N_MELS + m) * CHUNK_FRAMES;
            data[to..to + len].copy_from_slice(from);
        }
    }
    Features { data, chunks }
}

/// whisper.cpp's log-mel spectrogram: `(N_MELS, frames)`, row-major, where
/// `frames` is the sample count over the hop, rounded up to a half chunk,
/// plus one more half chunk.
pub fn log_mel_spectrogram(samples: &[f32], filters: &[f32]) -> Vec<f32> {
    let n_len = (samples.len() / HOP_LENGTH).div_ceil(PAD_FRAMES) * PAD_FRAMES + PAD_FRAMES;
    let mut padded = samples.to_vec();
    padded.resize(n_len * HOP_LENGTH, 0.0);

    #[allow(clippy::cast_precision_loss)]
    let hann: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - ((2.0 * PI * i as f32) / N_FFT as f32).cos()))
        .collect();

    // Each frame is independent of the others, so they are split across
    // threads in contiguous runs and scattered into place afterwards. Which
    // thread computes a frame does not change a bit of it.
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(12);
    let per_thread = n_len.div_ceil(threads);
    let runs: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_len)
            .step_by(per_thread)
            .map(|start| {
                let (hann, samples) = (&hann, &padded);
                let end = (start + per_thread).min(n_len);
                s.spawn(move || {
                    let mut out = Vec::with_capacity((end - start) * N_MELS);
                    for frame in start..end {
                        mel_frame(hann, samples, filters, frame, &mut out);
                    }
                    (start, out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a mel thread panicked"))
            .collect()
    });

    let mut mel = vec![0.0f32; N_MELS * n_len];
    for (start, run) in runs {
        for (offset, frame) in run.as_chunks::<N_MELS>().0.iter().enumerate() {
            for (m, &v) in frame.iter().enumerate() {
                mel[m * n_len + start + offset] = v;
            }
        }
    }

    let floor = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for m in &mut mel {
        *m = m.max(floor) / 4.0 + 1.0;
    }
    mel
}

/// Append the `N_MELS` log10 energies of frame `frame` to `out`.
fn mel_frame(hann: &[f32], samples: &[f32], filters: &[f32], frame: usize, out: &mut Vec<f32>) {
    let n_fft = 1 + N_FFT / 2;
    let offset = frame * HOP_LENGTH;
    let available = samples.len() - offset;
    let mut windowed = vec![0.0f32; N_FFT];
    for j in 0..N_FFT.min(available) {
        windowed[j] = hann[j] * samples[offset + j];
    }

    let mut power = fft(&windowed);
    for j in 0..N_FFT {
        power[j] = power[2 * j] * power[2 * j] + power[2 * j + 1] * power[2 * j + 1];
    }
    for j in 1..N_FFT / 2 {
        let mirror = power[N_FFT - j];
        power[j] += mirror;
    }

    for m in 0..N_MELS {
        let row = &filters[m * n_fft..(m + 1) * n_fft];
        let mut sum = 0.0f32;
        let mut k = 0;
        // Unrolled by four, as whisper.cpp does: the grouping is part of the
        // result, since float addition does not reassociate.
        while k < n_fft.saturating_sub(3) {
            sum += power[k] * row[k]
                + power[k + 1] * row[k + 1]
                + power[k + 2] * row[k + 2]
                + power[k + 3] * row[k + 3];
            k += 4;
        }
        while k < n_fft {
            sum += power[k] * row[k];
            k += 1;
        }
        out.push(sum.max(1e-10).log10());
    }
}

/// Recursive radix-2 FFT, falling back to a DFT on odd sizes. Returns
/// interleaved `re, im` pairs.
fn fft(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 1 {
        return vec![input[0], 0.0];
    }
    if n % 2 == 1 {
        return dft(input);
    }
    let even: Vec<f32> = input.iter().step_by(2).copied().collect();
    let odd: Vec<f32> = input.iter().skip(1).step_by(2).copied().collect();
    let even = fft(&even);
    let odd = fft(&odd);

    let mut out = vec![0.0; n * 2];
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f32;
    for k in 0..n / 2 {
        #[allow(clippy::cast_precision_loss)]
        let theta = (PI + PI) * k as f32 / n_f;
        let re = theta.cos();
        let im = -theta.sin();
        let (re_odd, im_odd) = (odd[2 * k], odd[2 * k + 1]);
        out[2 * k] = even[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + n / 2)] = even[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

/// The naive DFT. Returns interleaved `re, im` pairs.
fn dft(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f32;
    let mut out = Vec::with_capacity(2 * n);
    for k in 0..n {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (j, &x) in input.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let angle = (PI + PI) * k as f32 * j as f32 / n_f;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        out.push(re);
        out.push(im);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
    }

    /// The case candle's own tests pin, in f32.
    #[test]
    fn fft_of_an_impulse() {
        let out = fft(&[0.0, 1.0, 0.0, 0.0]);
        assert!(
            close(&out, &[1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0]),
            "{out:?}"
        );
    }

    #[test]
    fn fft_agrees_with_the_dft_it_falls_back_to() {
        #[allow(clippy::cast_precision_loss)]
        let input: Vec<f32> = (0..N_FFT).map(|i| (i as f32 * 0.37).sin()).collect();
        let (fast, slow) = (fft(&input), dft(&input));
        let worst = fast
            .iter()
            .zip(&slow)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-2, "fft and dft differ by up to {worst}");
    }

    #[test]
    fn mel_filters_decode_to_expected_count() {
        assert_eq!(
            MEL_FILTERS.len() % 4,
            0,
            "embedded mel blob must be f32-aligned"
        );
        let f = mel_filters();
        assert_eq!(f.len(), N_MELS * (N_FFT / 2 + 1));
        assert!(f.iter().all(|x| x.is_finite()));
    }

    /// A chunk of audio comes out as two chunks of features: whisper.cpp's
    /// extra half chunk tips the frame count past a single chunk.
    #[test]
    fn one_chunk_of_audio_is_two_chunks_of_features() {
        let samples = vec![0.0; CHUNK_SAMPLES];
        let mel = log_mel_spectrogram(&samples, &mel_filters());
        assert_eq!(mel.len(), N_MELS * (CHUNK_FRAMES + PAD_FRAMES));
        let features = extract_features(&samples, &mel_filters());
        assert_eq!(features.chunks, 2);
        assert_eq!(features.data.len(), 2 * N_MELS * CHUNK_FRAMES);
    }

    /// Past the spectrogram's frames the second chunk is zeros, not silence:
    /// the reshape pads with zeros, where silence normalizes to the floor.
    #[test]
    fn the_chunk_padding_is_zeros() {
        let samples: Vec<f32> = (0..CHUNK_SAMPLES)
            .map(|i| if i % 50 < 25 { 0.1 } else { -0.1 })
            .collect();
        let features = extract_features(&samples, &mel_filters());
        let second = &features.data[N_MELS * CHUNK_FRAMES..];
        for m in 0..N_MELS {
            let row = &second[m * CHUNK_FRAMES..(m + 1) * CHUNK_FRAMES];
            assert!(row[PAD_FRAMES..].iter().all(|&v| v == 0.0));
            assert!(row[..PAD_FRAMES].iter().all(|&v| v != 0.0));
        }
    }

    /// The layout is chunk-major: the first chunk's rows are the spectrogram's
    /// first `CHUNK_FRAMES` columns.
    #[test]
    fn features_are_the_spectrogram_cut_into_chunks() {
        #[allow(clippy::cast_precision_loss)]
        let samples: Vec<f32> = (0..CHUNK_SAMPLES)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let filters = mel_filters();
        let mel = log_mel_spectrogram(&samples, &filters);
        let frames = mel.len() / N_MELS;
        let features = extract_features(&samples, &filters);
        for m in [0, 17, N_MELS - 1] {
            for t in [0, 1, CHUNK_FRAMES - 1] {
                assert_eq!(
                    features.data[m * CHUNK_FRAMES + t].to_bits(),
                    mel[m * frames + t].to_bits()
                );
            }
            let t = 5;
            assert_eq!(
                features.data[(N_MELS + m) * CHUNK_FRAMES + t].to_bits(),
                mel[m * frames + CHUNK_FRAMES + t].to_bits()
            );
        }
    }

    #[test]
    fn pad_to_chunk_leaves_exact_multiples_unchanged() {
        let a = vec![0.5f32; 8];
        assert_eq!(pad_to_chunk(&a, 4), a);
    }

    #[test]
    fn pad_to_chunk_rounds_up_and_zero_fills() {
        let p = pad_to_chunk(&[1.0f32; 5], 4);
        assert_eq!(p.len(), 8);
        assert_eq!(&p[..5], &[1.0; 5]);
        assert_eq!(&p[5..], &[0.0; 3]);
    }

    #[test]
    fn pad_to_chunk_pads_sub_chunk_input_to_one_chunk() {
        assert_eq!(pad_to_chunk(&[1.0f32], 4).len(), 4);
    }

    #[test]
    fn pad_to_chunk_empty_stays_empty() {
        // 0 is a multiple of any chunk, so empty input yields ZERO chunks. The
        // transcribe handler rejects empty audio (400 invalid_audio) before it
        // can reach here; this pins the helper's own behavior at the boundary.
        assert!(pad_to_chunk(&[], 4).is_empty());
    }
}
