# elroy-rs

`elroy-rs` is a Rust rewrite of the Python `elroy` project at [`../elroy`](../elroy).

The goal is not to build a new product. The goal is to reproduce Elroy's current behavior in Rust with clearer boundaries, stronger type safety, and a more maintainable runtime.

## Current Status

This repository is in the workspace bootstrap phase.

Current implemented baseline:

- Cargo workspace with foundational crates
- standard `just` commands for format, lint, test, check, and run
- initial config, session/turn, and bootstrap data model types
- provider-neutral conversation orchestrator skeleton in `elroy-core`
- provider-neutral turn runner in `elroy-core` that accumulates both stream events and normalized transcript messages
- live provider adapter in `elroy-core` that can drive the orchestrator with real OpenAI or Anthropic HTTP clients
- reusable application runtime layer in `elroy-app` for shared prompt execution, snapshot loading, and DB-backed tool wiring
- first domain crates for file-backed repository behavior in `elroy-memory` and `elroy-agenda`
- tested recursive discovery of file-backed memory and agenda documents
- YAML config loading with `ELROY_*` environment override precedence plus provider API configuration
- initial `ratatui` crate with a tested two-pane Elroy app shell
- pure TUI focus/key-mode state machine derived from `../elroy/UI.md`
- runnable `ratatui` + `crossterm` terminal loop exposed via `just run --tui`
- canonical internal tool schema with OpenAI/Anthropic adapter projections
- initial provider-neutral LLM stream event model and tool-call accumulator
- OpenAI- and Anthropic-style request payload builders in `elroy-llm`
- normalized context-message model with role/tool-call validation in `elroy-llm`
- live HTTP model clients for OpenAI Responses and Anthropic Messages APIs with mock-backed tests
- SQLite-first persistence direction encoded in `elroy-db`
- SQLite migrations plus persisted bootstrap inventory and derived `memories` / `agenda_items` tables rebuilt from compatible files
- SQLite-backed context-message persistence for the local conversation transcript
- DB-backed executable tools for memory listing/search/show/update/archive, agenda listing/show/create/update/complete/delete plus agenda checklist add/edit/complete, and due-item create/list/update/rename/complete/delete in the live prompt path
- TUI loading from persisted transcript, memory, and agenda data plus live prompt submission through the shared runtime
- DB-backed sidebar item opening for memory and agenda detail inspection in the TUI
- first TUI sidebar mutation hotkeys backed by the shared runtime: `d` archives a selected memory, while `c`/`d` complete or delete a selected agenda item
- agenda reads now use an agenda-only view, while trigger-based reminder items are surfaced separately through due-item tools
- CLI `--prompt` path for direct live model queries through the configured provider

Before substantial code lands, the repository uses these documents as the source of truth:

- `AGENTS.md` for coding-agent instructions and repo workflow rules
- `REWRITE_PLAN.md` for migration scope, order, and milestones
- `PARITY_MATRIX.md` for feature tracking against the Python implementation
- `ARCHITECTURE.md` for Rust crate and dependency boundaries
- `TEST_STRATEGY.md` for parity-validation expectations

## Rewrite Goals

- Preserve user-visible Elroy behavior unless a change is explicitly documented
- Port the current Python architecture intentionally instead of redesigning during implementation
- Keep migration progress measurable at the subsystem and behavior level
- Build a test strategy that proves parity, not just compilation
- Preserve Elroy's inspectable file-backed knowledge model where the Python app exposes recallable/user-visible content through files
- Allow `elroy-rs` to bootstrap its derived database/index state from the same file set used by the Python implementation

## Non-Goals

- Reimagining product scope during the rewrite
- Introducing speculative features before parity
- Replacing every Python implementation detail one-for-one when Rust needs a different mechanism
- Moving user-inspectable recallable content behind opaque database-only storage without an explicit migration decision

## Source Of Truth

The current implementation lives in `../elroy`.

Important source areas:

- `../elroy/elroy/core/` for session, runtime, and orchestration boundaries
- `../elroy/elroy/ui/` for TUI behavior
- `../elroy/elroy/tools/` for tool registration and execution
- `../elroy/elroy/repository/` for domain logic
- `../elroy/elroy/db/` for persistence and migrations
- `../elroy/tests/` for behavioral expectations
- `../elroy/UI.md` for keyboard and TUI interaction behavior
- `../elroy/ROADMAP.md` and `../elroy/docs/` for product and feature context

## Working Rules

- Update `PARITY_MATRIX.md` when a subsystem changes state
- Update `REWRITE_PLAN.md` when milestone scope or sequencing changes
- Document intentional deltas from Python before merging them
- Prefer adding tests and fixtures alongside the ported behavior

## Current Commands

- `just run` runs the bootstrap and DB rebuild path against the configured Elroy home
- `just run -- --tui` launches the current `ratatui` shell
- the TUI now submits prompts through the live runtime and can open sidebar entries into the conversation pane
- `just run -- --prompt "hello"` sends a direct live query through the configured model provider

The live query path uses:

- `OPENAI_API_KEY` for OpenAI-backed models
- `ANTHROPIC_API_KEY` for Claude-backed models
- `ELROY_OPENAI_BASE_URL` and `ELROY_ANTHROPIC_BASE_URL` for endpoint overrides
- `ELROY_ANTHROPIC_API_VERSION` for Anthropic API-version overrides

## Early Implementation Priorities

1. Bootstrap the Rust workspace and toolchain conventions
2. Define core runtime/session/config boundaries
3. Establish test harnesses and parity fixtures
4. Port one subsystem at a time in dependency order

## Near-Term Deliverables

- Rust workspace skeleton
- Initial CLI entrypoint
- Core config/session/runtime types
- First parity tests around config and command surface
- Bootstrap path that can ingest existing Elroy files and reconstruct required derived state

Current gaps:

- no streaming live model integration yet
- no full domain DB schema or higher-level repository workflows yet
- no broad repository-backed mutation workflows yet beyond the initial file-backed memory and agenda operations now split into `elroy-memory` and `elroy-agenda`
- no Codex workflow implementation yet
