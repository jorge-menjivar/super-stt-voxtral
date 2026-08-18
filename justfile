# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone Voxtral backend. Mirrors the recipe names used
# by the main super-stt repo (`just check`, etc.).

# Default: build release
default: build-release

# Compiles with debug profile. Usage: just build-debug [--features cuda]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile. Usage: just build-release [--features cuda]
build-release *args:
    cargo build --release --locked {{ args }}

# Build, then copy the binary to the entrypoint name `backend.toml` declares, so
# this directory can be installed with the daemon's Import-from-dir path. Cargo
# already names the artifact that, and the release workflow tarballs it under the
# same name, so a local install and a published one stage the same bytes under
# the same name. Usage: just stage [--features cuda]
stage *args: (build-release args)
    cp target/release/super-stt-backend-voxtral super-stt-backend-voxtral
    @echo "staged super-stt-backend-voxtral — this directory is now installable with Import from dir"

# Runs a clippy check — mirrors super-stt's lint. There, `--all-features
# --workspace` enables no CUDA (workspace crates have no cuda feature; the GPU
# backends are out-of-tree), so the equivalent here is a default-feature (CPU)
# lint, which still covers all of voxtral's own code. Run `just check
# --all-features` locally to additionally lint the candle CUDA backend (needs a
# CUDA toolkit).
check *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Runs a clippy check with JSON message format (consumed by clippy-sarif in CI)
check-json: (check '--message-format=json')

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

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
