# Open Questions

This document tracks rewrite questions that still need explicit decisions.

Resolved questions should be removed from here once the decision is recorded in the canonical design docs.

## Open

### Workspace / Crate Split

Question:

- Should the rewrite begin with many focused crates or start closer to a 1:1 port and split later?

Why it matters:

- too much early splitting may slow the port
- too little structure may let boundaries collapse

Current direction:

- either approach is acceptable
- a close 1:1 port is acceptable initially if responsibility boundaries stay explicit

## Resolved Direction

These are no longer open at the principle level:

- `ratatui` is the selected TUI framework
- async-first runtime model
- SQLite-first persistence is the target, with `rusqlite` plus `refinery` as the working direction and `tokio-rusqlite` available if async DB boundaries need it
- file-backed inspectable recallable content where Python uses files as visible source of truth
- ability to start from compatible Elroy files and backfill/rebuild derived DB state
- internal canonical tool schema with provider-specific adapters is the selected tool compatibility strategy
- Codex workflow support required for the first Rust version
- strong reliance on tests with explicit test documentation
- test-layer expectations are defined in `TEST_STRATEGY.md`
