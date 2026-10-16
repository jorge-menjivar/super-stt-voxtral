# Notes on the Burn port

The measurements and findings behind choices in the code. The code's comments
say what it does and why; the numbers, and how they were found, are here.

## Precision

The backend runs in f16 on every accelerator (`inference::model_dtype`).

- **f16 against bf16.** On CUDA, the parity test puts f16 at candle's own f16
  drift through every layer, or under it: 0.398 relative error at the last
  decoder layer, where candle's f16 is at 0.399. bf16, the dtype the
  checkpoints ship in, sits 1.2–3x above candle's f16 on the same rows: the
  three mantissa bits it gives up. Its extra range buys nothing here, since
  Voxtral's largest activations are in the hundreds.
- **bf16 in CubeCL.**
  - For AMD, CubeCL compiles through LLVM, and its LLVM backend has no bf16
    type (`cubecl-llvm`, `to_llvm/ty.rs`). Every kernel that uses one fails to
    compile, on any AMD card.
  - SPIR-V allows bf16 only in conversions, dot products and cooperative
    matrices, yet CubeCL compiles bf16 arithmetic for Vulkan. NVIDIA's 610.57
    driver segfaults on it.
  - CubeCL's Metal backend registers no bf16 at all.
  - `supports_dtype` reports bf16 as supported on the first two all the same.
- **A failed warm-up fails the load** because of the AMD case. A bf16 kernel
  failed to compile during the warm-up, the backend swallowed the error,
  reported `ready`, and then failed every transcription.
- **f32** weights alone are 19 GB; a 24 GB card only runs through by retrying
  allocations that ran out of memory.
- **f32 on a GPU is not exact.** The first convolution sits 2e-4 from candle's,
  about where TF32 tensor cores would put it, which is why the parity test
  holds f32 to candle's f16 drift. The CPU build the port was first checked
  on, since dropped, had bit-identical features and sat within 7e-4 of candle
  at every layer.

## Loading

- burn-store's `FloatCastAdapter` converts one element at a time on one thread.
  For Voxtral Mini's 4.7 billion weights that added 8 s to every f16 load.
  `HalfCast` does the same arithmetic on every core; its test checks all 65,536
  bf16 bit patterns against burn's conversion.
- On the AMD BC-250, reading the weights takes about 5 minutes of every load.

## Memory

- **One stream.** CubeCL gives each thread its own stream and each stream its
  own memory pools. tokio's blocking pool retires idle threads after ten
  seconds, so requests further apart than that each landed on a new thread and
  new pools: 2.3 GB more per request until the card ran out. Hence
  `StreamPolicy::Single`.
- **Releasing working memory.** An idle backend held its largest request's
  working memory, over 2 GB on top of the weights. Handing it back takes about
  70 ms, and the next request about 50 ms more to allocate it again, which is
  why the release runs after the reply.

## Decoder

- Linear weights are kept column-major. The Burn port of Qwen3-TTS measured
  the one-row matmul of a decoding step at about 55 µs with a row-major weight
  and about 12 µs, the memory-bandwidth limit, with a column-major one, on an
  RTX 3090.

## Kernel cache and load progress

The warm-up's progress is measured in kernel-cache entries, against a
per-accelerator estimate (`WARM_UP_CACHE_ENTRIES`). Each count is from a first
load into an empty cache with Voxtral Mini, excluding the throughput probes.

| Accelerator | Burn `1e9de733` / CubeCL `46f7861b` | Burn `059c75e2` / CubeCL `8f4b1b36` |
|---|---|---|
| CUDA, RTX 3090 | 714: 42 tuning results, 672 kernel entries | 1166: 44 tuning results and 359 kernels, which CUDA writes twice, each with a record, plus one session record |
| Vulkan, RTX 3090 | 286: 42 and 244 | 553: 44 tuning results and 232 kernels, each with a record, plus one session record |
| ROCm, AMD BC-250 | about 186 | about 351 |
| Metal | not measured; uses Vulkan's count | same |

The ROCm counts are read off the last progress sample before `ready`, through
the estimate's inverse, since the cache also holds what the transcriptions
after it wrote. The same reading reproduces the CUDA and Vulkan counts within a
few percent. Before they were measured, ROCm was assumed to write what CUDA
does, and its bar ended at 26%.

The daemon fails a load whose step and progress stand still for 120 s. The
longest stretch without a change on a first load:

| Accelerator | Old pins | New pins |
|---|---|---|
| CUDA, RTX 3090 | 2.6 s | 2.3–2.6 s |
| Vulkan, RTX 3090 | 1.6 s | 1.8–2.5 s |
| ROCm, AMD BC-250 | 62.6 s | 105.8 s |

On ROCm the long stretch falls during `building_kernels`, about a quarter of the
way in. CubeCL compiles through LLVM there, on a slow CPU, and writes no cache
entry for that long; its environment API exposes nothing finer to measure.
super-stt-qwen saw 107 s on the same board and pins.

The warm-up covers clips up to 30 s. The first longer clip after a first load
compiles and tunes its own shapes: 70–80 s for a 34 s clip on CUDA, about
23 s on Vulkan, and 15–18 s more than later runs on the BC-250.

## Building the kernels before the weights arrive

`POST /v1/prepare` (contract v4) builds the kernels from the two files the
manifest lists as the model's `prepare_files`, `config.json` and
`tekken.json`. The daemon can then run it while the weights download. It runs
the load's warm-up on a model with two encoder layers, two decoder layers and
every weight zero. A load sent while it runs waits for it, then maps the
weights. Both models list `prepare_files`. Voxtral Small's prepare has not
been run, nor has its real load, which needs about 52 GB.

Measured 2026-10-16 on the RTX 3090 with Voxtral Mini. Each run started from an
empty kernel cache in one backend process, driven the way the daemon drives
it: either a load once the status read `prepared` ("after"), or a load sent
20 s into the prepare ("during").

| | CUDA | Vulkan |
|---|---|---|
| prepare | 253 s | 82 s |
| load after it | 9.8 s, phase `loading` | 11.0 s, phase `loading` |
| load sent during it, ready at | 240 s | 91 s |
| plain first load, for comparison | 241–247 s | 84–97 s |
| peak during the prepare | 5.4–5.5 GiB | 5.1 GiB |
| left when the prepare hands over | 0.4–0.7 GiB | 0.1–0.3 GiB |
| peak of the process, transcriptions included | 10.2 GiB | 9.6 GiB |
| peak of a plain first load | 11.8–12.0 GiB | 11.4 GiB |
| longest stretch without progress | 3.2 s (14.2 s with a build on the CPU) | 2.9–3.2 s |

- **Kernels left to the load.** The load after a prepare compiled one more
  kernel, an element-wise fusion, and its warm-up took 0.6–0.9 s.
- **Total time.** Preparing does not shorten a first load done in one go. The
  gain is that the build no longer waits for the 9 GB of weights. The peak
  also drops by 1.6–1.8 GiB, because the zeroed model's memory goes back to
  the driver before the weights map. On CUDA the process's peak came after
  the load, with the first 34 s clip.
- **Progress during an overlap.** A load sent during the prepare keeps its
  progress and reports one `initial_setup`: `building_kernels`, then
  `loading_weights` and `warming_up`.
- **Kernel-cache entries.** A prepare in its own process, an earlier version
  of the same build, wrote 1183 entries on CUDA and 553 on Vulkan. That is
  what a first load writes, so `WARM_UP_CACHE_ENTRIES` measures it as well.
- **After a restart.** A second process over the same cache answers `prepared`
  at once, from the warm marker.
- **Clips over 30 s.** The first such clip still compiles and tunes its own
  shapes, as after a plain first load.

## Moving to Burn and CubeCL main

From Burn `1e9de733` (`qwen3-tts`) and CubeCL `46f7861b`
(`compilation-cache-default`) to Burn `059c75e2` (`qwen3-tts-main`) and CubeCL
`8f4b1b36` (`compilation-cache-main`). Measured 2026-10-14 and 15.

- **On the RTX 3090:** the two sides were alternated, each on two fresh kernel
  caches, because one tuning run can lock in a slower kernel for a whole cache.
  The card was also driving a desktop, so timings vary by up to about 20%
  between runs.
- **On the BC-250:** one cache each, run one after the other.
- **Memory** is the process's GPU memory from `nvidia-smi`. On the BC-250 it is
  the GTT in use by every process, of which about 80 MiB was not the backend.

| | CUDA | Vulkan | ROCm (BC-250) |
|---|---|---|---|
| first load | 220–228 → 241–247 s | 83–84 → 84–97 s | 638 → 736 s |
| later load | 7.9–8.2 → 8.0–8.2 s | 8.9–9.3 → 9.3–9.5 s | 336 → 334 s |
| peak of a first load | 16.2 → 11.8–12.0 GiB | 16.1 → 11.4 GiB | 11.7 → 11.7 GiB |
| peak of a later load | 11.2 → 9.6 GiB | 11.1 → 9.1 GiB | 10.0 → 9.8 GiB |
| peak during a transcription | 11.4 → 9.9 GiB | 11.2 → 9.3 GiB | |
| idle | 9.0–9.15 → 9.2 GiB | 8.8 → 8.8 GiB | 9.4–9.5 → 9.4 GiB |
| 11 s clip | 0.84–1.00 → 0.85–0.89 s | 0.94–0.99 → 0.92–1.00 s | 26.8–27.2 → 26.9–27.2 s |
| 34 s clip | 1.9–2.3 → 1.9–2.0 s | 2.1–3.4 → 2.1–2.3 s | 46.6–46.9 → 46.5–47.2 s |

- **Transcripts** were word for word the same on every run.
- **CUDA parity** on the new pins: 0.3981 at the last decoder layer, against
  candle's f16 at 0.3988, with the greedy tokens matching at all 27 steps.
- **Autotune on the 3090** picks the new pins' im2col convolution
  (`conv_im2col`) for all four of Whisper's convolution shapes on Vulkan, where
  the old pins ran `simple_sync_cmma`. CUDA keeps its tensor-core kernels,
  `simple_sync_mma`, `simple_sync_cmma` and `simple_async_*mma`, on both. The
  convolutions are too small a share of a transcription for the change to
  show in its time.
- **The CUDA build uses CubeCL's C++ backend.** With LLVM, CubeCL's new
  default, super-stt-qwen measured its transcriptions at half speed on the
  same card. Voxtral was measured with the C++ backend only.
- **ROCm keeps LLVM.** super-stt-qwen found that CubeCL's C++ backend for HIP
  left a first load on the BC-250 without progress for over four minutes.
- **The new pins compile no CPU backend into the GPU builds.** The old ones
  always carried `burn-flex`.

## ROCm on the AMD BC-250

- **Patched driver.** The board needs its patched amdgpu module, which turns
  off `flush_pasid_uses_kiq` (ROCm issue 6313). With the stock module, which a
  kernel update brought back on 2026-10-05, a Voxtral load hung the GPU: the
  driver logged `timeout waiting for kiq fence`, then a gfx ring timeout. With
  `gpu_recovery=0`, only a reboot brought the GPU back. Both pin sets hung that
  way, and both ran clean once the module was rebuilt for the new kernel.
- **Compiler logging.** CubeCL's ROCm compiler, pliron, logs its whole IR after
  every pass at `info`. One warm-up wrote a 10 GB `backend.log`, which is why
  `main` starts pliron at `warn`.

## Provenance of details kept from the candle backend

- `MAX_NEW_TOKENS`, 1000, is the cap the candle backend decoded to.
- `post_process_transcription` is the candle backend's, verbatim.
- `lang:en` encodes to `[9909, 1058, 1262]`, the ids the candle backend
  hard-coded. The whole prompt matches the one it built, id for id.
- The log-mel features are bit-for-bit what the candle backend sent the model,
  including the two choices that differ from the Hugging Face feature
  extractor (see `voxtral::audio`).
- The candle port read `tie_word_embeddings` out of the `attention_bias` key;
  this port reads each from its own.
