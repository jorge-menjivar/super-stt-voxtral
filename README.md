# Super STT — Voxtral backend

[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/super-stt-voxtral/coverage.json)](https://jorge-menjivar.github.io/super-stt-voxtral/)

A speech-to-text backend for **[Super STT](https://github.com/jorge-menjivar/super-stt)**.
It runs [Mistral's Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
models locally on your GPU to turn speech into text.

Super STT is an on-device speech-to-text engine. It doesn't ship any models of
its own — it loads **backends** like this one at runtime. This repo packages the
Voxtral models (Voxtral Mini 3B and Voxtral Small 24B) as one of those backends.

## Using it

You don't run this directly. Super STT discovers it through its backend
registry, downloads a prebuilt release for your platform, fetches the model
weights, and runs it sandboxed. To use Voxtral, install Super STT and enable it
from the app — see the [Super STT docs](https://github.com/jorge-menjivar/super-stt).

## Models

Two models, chosen by `name` when Super STT loads the backend. Both want a GPU
— NVIDIA through CUDA, AMD through ROCm, or anything else through Vulkan; the
weights are pulled from Hugging Face on first load.

| Model (`name`)           | Upstream model                                                               | ~VRAM  |
| ------------------------ | ---------------------------------------------------------------------------- | ------ |
| `voxtral-mini-3b-2507`   | [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)     | ~12 GB |
| `voxtral-small-24b-2507` | [Voxtral Small 24B](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | ~52 GB |

## What's in here

A small, self-contained Rust program that loads a Voxtral model and speaks the
Super STT backend protocol (a tiny HTTP API over a Unix socket). It shares no
code with the Super STT project.

The model runs on [Burn](https://github.com/tracel-ai/burn). Its GPU kernels are
compiled at runtime by CubeCL, so one binary per accelerator covers every GPU
generation the driver can compile for — which is also what reaches ROCm and
Vulkan. The port lives in `src/voxtral`:

| Path | What it holds |
|---|---|
| `src/voxtral/audio.rs` | 16 kHz samples to the log-mel spectrogram, cut into 30-second chunks. |
| `src/voxtral/encoder.rs` | The audio tower (Whisper's encoder) and the projector into the decoder. |
| `src/voxtral/transformer.rs` | The text decoder, a Llama stack. |
| `src/voxtral/model.rs` | Loading the sharded checkpoint, splicing the audio into the prompt, greedy decoding. |
| `src/inference.rs` | Device selection, the prompt, and the transcript cleanup. |
| `parity/` | The candle reference the port is checked against, see below. |

### Performance

Measured on an RTX 3090 against the candle build this backend shipped before
(v0.1.1), with requests back to back:

| | candle (f16) | Burn (bf16) |
|---|---:|---:|
| 11-second clip (27 tokens) | 0.91–0.93 s | 0.77–0.88 s |
| 34-second clip (79 tokens) | 2.29–2.31 s | 1.71–1.80 s |
| first transcription after a load | 8.4 s | 0.8 s |
| load, kernel cache warm | 3.2 s | 4.5–5.9 s |
| load, first ever on the machine | 3.2 s | 4–5 min |
| VRAM, idle | 9.8–10.7 GiB | 9.0 GiB |
| VRAM, during a transcription | 10.7 GiB | 11.3 GiB |

A request after a pause takes longer on both, 1.1–1.3 s for the short clip on
candle and 0.8–1.1 s on Burn: the GPU drops to its idle clocks in between.
Burn hands each transcription's working memory back once it has replied, which
is what keeps it idle at little more than its weights; the next request
allocates it again, for about 50 ms.

The decoder's key/value caches are allocated up front and every decoding step
attends to all of them through a mask, so every step runs the same kernels on
the same shapes: a cache grown a token at a time gave CubeCL a new shape to
compile and tune for every token, and the first transcription after a load
took 75 seconds.

### Kernel cache

CubeCL compiles the kernels a transcription needs the first time it meets them
and tunes each operation for its shape, so a load ends with a warm-up
transcription of silence: the first real request then finds everything ready.
The warm-up runs at the shapes of a clip under 30 seconds. A longer one meets
new shapes, and the first such clip on a machine compiles and tunes them,
about 70 seconds for a 34-second clip, before the cache holds them too.
Compiled kernels are kept in `SUPER_STT_BACKEND_CACHE_DIR` when the daemon
grants one — everything else the sandbox exposes is read-only — so only the
first load of a build pays; without it the backend logs a warning and
recompiles on every load.

## Building from source

Most people never need to — Super STT downloads prebuilt releases. For
development (requires [`just`](https://github.com/casey/just)):

```bash
just build-release   # the pure-Rust CPU backend
just build-cuda      # needs the CUDA headers — no GPU, no compute capability
just build-rocm      # needs the ROCm headers
just build-vulkan    # needs nothing; the loader is found at runtime
just ci              # format, lint, build, and test
```

Each build carries exactly one accelerator, which is why the GPU recipes pass
`--no-default-features`: cargo features are additive, so `--features cuda` alone
would keep the default CPU backend too.

**Burn comes from a fork.** `Cargo.toml` pins `jorge-menjivar/burn` on the
branch the Qwen TTS backend builds on, which carries fixes to Burn's fusion
crates and to tensor reads that are not upstream yet, along with the matching
CubeCL patch.

## Checking the port against candle

This backend ran on candle before it ran on Burn. `parity/` is a small separate
crate — its own workspace, so candle never enters this backend's build — that
runs candle's Voxtral over `tests/data/jfk.wav` and writes every layer's output
to a file. The test in `src/voxtral/parity.rs` runs the port over the same
inputs and prints, for each of the ~100 taps (the features, both convolutions,
the 32 encoder layers, the projector, the 30 decoder layers, and the logits of
every decoding step), the largest absolute difference, the relative L2 error
and the cosine distance:

```bash
export SUPER_STT_BACKEND_DIR=<dir holding models/voxtral-mini-3b-2507>
just parity                                              # CPU, f32
just parity bf16 --no-default-features --features cuda   # GPU, bf16
```

What it measured on this port, every tap against candle's f32:

- **f32 on the CPU**: the features are bit-identical, every layer is within
  7e-4 relative error, and the tokens match. The test holds it to 1e-3.
- **f16 on CUDA**: every layer drifts as much as candle's own f16 build does
  (0.398 against 0.399 at the last decoder layer) or less.
- **bf16 on CUDA** — what ships: 1.2–3x candle f16's drift, which is the three
  mantissa bits bf16 gives up; the tokens still match. The deep-decoder drift
  in both 16-bit types comes from activations in the hundreds, not the port.

A GPU picks bf16 when it can compute in it and f16 otherwise: Vulkan often
stores bf16 without any arithmetic on it.

## License

GPL-3.0-only.
