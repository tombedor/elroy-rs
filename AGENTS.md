# Agent Instructions For elroy-rs

This repository is a controlled rewrite of `../elroy`, not a greenfield assistant project.

## Primary Objective

Ship a Rust implementation of Elroy that reaches behavioral parity with the Python project in incremental, testable stages.

When in doubt, preserve behavior first and improve internals second.

## Source Of Truth

Use these sources in order:

1. This repository's docs
2. `../elroy/tests/`
3. `../elroy/UI.md`
4. `../elroy/elroy/`
5. `../elroy/docs/` and `../elroy/ROADMAP.md`

If the Rust implementation intentionally diverges from Python, record it in `PARITY_MATRIX.md` and the relevant design doc before or with the code change.

## Required Workflow

- Read the relevant Python source before implementing a subsystem
- Identify the user-visible behavior and invariants first
- Implement the smallest coherent vertical slice possible
- Add or update tests in the same change
- Update `PARITY_MATRIX.md` status for the touched subsystem
- Update `REWRITE_PLAN.md` when milestone progress materially changes

## Rewrite Guardrails

- Do not redesign product behavior during parity phases unless explicitly requested
- Do not silently omit Python behavior because it is inconvenient to port
- Do not add speculative abstractions before a concrete need exists
- Do not copy Python naming blindly when Rust needs a more idiomatic surface, but preserve responsibility boundaries
- Do not treat "compiles" as "ported"

## Architecture Rules

Use `ARCHITECTURE.md` as the source of truth for module roles and dependency rules.

In particular:

- Workflow logic belongs in orchestrator-style components, not persistence modules
- Persistence modules should stay narrow and domain-scoped
- Builders/formatters should stay pure or mostly pure
- UI code must not absorb domain workflows
- App-wide composition belongs at the boundary, not in inner modules

## Validation Expectations

Before calling a change ready:

- Run formatting
- Run linting
- Run tests relevant to the touched subsystem
- Run the full standard validation command if it exists for the workspace

If the Rust command surface is not created yet, create or update repo docs to define the intended commands before broad implementation continues.

## Expected Repository Conventions

Until the workspace is fully scaffolded, prefer these conventional targets:

- `cargo fmt`
- `cargo clippy --all-targets --all-features`
- `cargo test`

This repository now has a `justfile`. Prefer:

- `just fmt`
- `just lint`
- `just test`
- `just check`
- `just run`

## Definition Of Done For A Ported Slice

A slice is not done unless all of the following are true:

- The Rust code implements the intended behavior
- Tests cover the primary behavior and edge cases
- The parity state is updated
- Known deviations are explicitly documented
- Follow-on work is recorded if parity is partial

## High-Risk Areas

Treat these areas as parity-sensitive:

- TUI keybindings, focus transitions, and streaming behavior
- Tool schema generation and tool execution loops
- Memory recall and consolidation behavior
- Reminder and due-item surfacing
- Config precedence and defaults
- Persistence boundaries and migration compatibility

## Review Standard

Default review posture:

- look for behavior regressions first
- look for boundary violations second
- look for missing parity coverage third

Do not accept "we can add parity later" unless the missing behavior is clearly marked as incomplete in `PARITY_MATRIX.md`.
