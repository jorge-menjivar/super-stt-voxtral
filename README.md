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

## What's in here

A small, self-contained Rust program that loads a Voxtral model and speaks the
Super STT backend protocol (a tiny HTTP API over a Unix socket). It shares no
code with the Super STT project.

## Building from source

Most people never need to — Super STT downloads prebuilt releases. For
development (requires [`just`](https://github.com/casey/just)):

```bash
just build-release                  # CPU build
just build-release --features cuda  # GPU build (needs a CUDA toolkit)
just ci                             # format, lint, build, and test
```

## License

GPL-3.0-only.
