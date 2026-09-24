# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone Voxtral backend. Mirrors the recipe names used
# by the main super-stt repo (`just check`, etc.).
#
# The model runs on Burn, pinned by revision to a fork in Cargo.toml. The first
# build fetches and compiles it, which is slow; nothing else about the build is
# unusual and there is no C toolchain to install.
#
# Every build carries one GPU accelerator; there is no CPU build. Vulkan is the
# default, since it needs no SDK. The other recipes pass `--no-default-features`
# with theirs named, matching the release workflow: features are additive, so a
# build that kept the default `vulkan` alongside `cuda` would carry two
# backends.

parity_clip := justfile_directory() / "tests/data/jfk.wav"
parity_dir := justfile_directory() / "target/parity"

# Default: build release
default: build-release

# Compiles with debug profile. Usage: just build-debug [args]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile — the Vulkan build, the default.
# Usage: just build-release [args]
build-release *args:
    cargo build --release --locked {{ args }}

# Build with CUDA. Needs the CUDA toolkit headers on the host; no GPU and no
# compute capability are needed, since CubeCL compiles the kernels at runtime.
build-cuda *args:
    cargo build --release --locked --no-default-features --features cuda {{ args }}

# Build with ROCm. Needs the ROCm headers `cubecl-hip-sys` binds against.
build-rocm *args:
    cargo build --release --locked --no-default-features --features rocm {{ args }}

# Build with Vulkan — the vendor-neutral GPU path. Needs no SDK to build; the
# loader is found at runtime.
build-vulkan *args:
    cargo build --release --locked --no-default-features --features vulkan {{ args }}

# Build with Metal, on macOS. Needs nothing beyond the Xcode command-line
# tools: wgpu reaches Metal through the system framework, and CubeCL compiles
# the kernels to MSL at runtime.
build-metal *args:
    cargo build --release --locked --no-default-features --features metal {{ args }}

# Build, then copy the binary to the entrypoint name `backend.toml` declares, so
# this directory can be installed with the daemon's Import-from-dir path. Cargo
# already names the artifact that, and the release workflow tarballs it under the
# same name, so a local install and a published one stage the same bytes under
# the same name. Usage: just stage [--no-default-features --features cuda]
stage *args: (build-release args)
    cp target/release/super-stt-backend-voxtral super-stt-backend-voxtral
    @echo "staged super-stt-backend-voxtral — this directory is now installable with Import from dir"

# Remove build output, the candle reference's included.
clean:
    cargo clean
    cd parity && cargo clean

# Runs a clippy check — mirrors super-stt's lint. Default features lint the
# Vulkan build, which compiles every line of the backend's own code but the
# other accelerators' arms of `select_device`; they differ only in which Burn
# feature is on.
check *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Runs a clippy check with JSON message format (consumed by clippy-sarif in CI)
check-json: (check '--message-format=json')

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all
    cd parity && cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check
    cd parity && cargo fmt --all -- --check

# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

# Dump every layer of candle's Voxtral over the test clip on the CPU, for
# `parity` to compare against: in f32, the reference, and in f16, the
# precision the candle backend shipped in and the yardstick for the port's
# own reduced precision. Needs the weights under
# $SUPER_STT_BACKEND_DIR/models/voxtral-mini-3b-2507. About eight minutes and
# 40 GB of RAM (candle's model runs twice per dump, to check the tapped copy
# against it); a dump that already exists is kept.
parity-reference:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${SUPER_STT_BACKEND_DIR:?set SUPER_STT_BACKEND_DIR to a directory holding models/voxtral-mini-3b-2507}"
    (cd parity && cargo build --release --locked)
    mkdir -p "{{ parity_dir }}"
    for dtype in f32 f16; do
        out="{{ parity_dir }}/candle-$dtype.safetensors"
        if [ -f "$out" ]; then echo "already present: $out"; continue; fi
        parity/target/release/voxtral-candle-reference \
            "$SUPER_STT_BACKEND_DIR/models/voxtral-mini-3b-2507" "{{ parity_clip }}" "$out" --dtype $dtype
    done

# Compare the Burn port against candle layer by layer and print the table,
# with candle's own f16 drift beside every row. `dtype` is the port's: f16 by
# default, which every accelerator runs; f32 is the strict check, and its 19 GB
# of weights want a 24 GB card.
# Usage: just parity [f16|bf16|f32] [--no-default-features --features cuda]
parity dtype="f16" *args: parity-reference
    SUPER_STT_PARITY_REF="{{ parity_dir }}/candle-f32.safetensors" \
    SUPER_STT_PARITY_BASELINE="{{ parity_dir }}/candle-f16.safetensors" \
    SUPER_STT_PARITY_DTYPE={{ dtype }} \
        cargo test --release --locked {{ args }} layers_match_candle -- --nocapture

# Measure code coverage (requires cargo-llvm-cov). --remap-path-prefix keeps the
# report paths relative (src/...), and tests/ is excluded so only product code
# is counted. Usage: just coverage [--html]
coverage *args:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex 'tests/'

# Full local CI gate: format, lint, build, test
# (no doctests — this is a binary-only crate, so `cargo test --doc` has no lib target)
ci: fmt-check check build-release test
