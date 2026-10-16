# Notes on the Burn port

The measurements and findings behind choices in the code. The code's comments
say what it does and why; the numbers, and how they were found, are here.

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
