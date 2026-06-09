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

# Measure code coverage (requires cargo-llvm-cov). Usage: just coverage [--html]
coverage *args:
    cargo llvm-cov --locked {{ args }}

# Coverage for CI: write lcov.info and print a summary
coverage-lcov:
    cargo llvm-cov --locked --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only

# Full local CI gate: format, lint, build, test
# (no doctests — this is a binary-only crate, so `cargo test --doc` has no lib target)
ci: fmt-check check build-release test
