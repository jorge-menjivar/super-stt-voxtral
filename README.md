# Super Voice — Voxtral backend

[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/super-stt-voxtral/coverage.json)](https://jorge-menjivar.github.io/super-stt-voxtral/)

Mistral's [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
speech recognition models, Voxtral Mini 3B and Voxtral Small 24B, as a
subprocess backend for [Super Voice](https://github.com/jorge-menjivar/super-voice).
Eight languages, on Linux with any GPU that CUDA, ROCm or Vulkan can drive, and
on Apple Silicon through Metal. It applies your dictation vocabulary: a term
it writes differently, or misspells in its vowels, comes out the way you spell
it.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single Rust binary that runs the models
through [Burn](https://github.com/tracel-ai/burn) and answers
`POST /v1/transcribe`.

## License

GPL-3.0-only. See `LICENSE`. The Voxtral weights the daemon fetches at runtime
are Mistral's, under Apache-2.0.
