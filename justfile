fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace

check:
    cargo check --workspace

run *ARGS:
    cargo run -p elroy-cli -- {{ARGS}}

help:
    @just --list
