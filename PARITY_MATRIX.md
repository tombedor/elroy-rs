# Parity Matrix

This document tracks rewrite progress against the Python implementation in `../elroy`.

## Status Vocabulary

- `not started`: no Rust implementation exists yet
- `spike`: exploratory code exists but is not a committed port
- `stubbed`: structure exists but behavior is mostly absent
- `partial`: meaningful behavior is implemented, but parity is incomplete
- `parity`: intended current behavior is implemented and tested
- `intentional delta`: behavior differs on purpose and the delta is documented

## Usage Rules

- Update this file in the same change as status-moving code
- Link tests or note missing coverage when a subsystem moves forward
- Record intentional deltas explicitly instead of hiding them under `partial`

## System-Level Tracking

| Python area | Behavior summary | Rust target | Status | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| `elroy/__main__.py` | package entrypoint into app | `crates/elroy-cli` and `crates/elroy-app` | `partial` | `crates/elroy-cli` smoke run via `just run` | Current Rust entrypoint performs config load plus DB bootstrap, and the shared `elroy-app` runtime now backs live prompt execution and TUI runtime actions |
| `elroy/config/` | env/config-file/model config resolution | `crates/elroy-config` | `partial` | `crates/elroy-config/src/lib.rs` | Defaults, YAML loading, and env override precedence exist; model/config surface is still narrow, though prompt execution now consults persisted assistant naming on top of config defaults |
| `elroy/core/` | session, turn, runtime, orchestration | `crates/elroy-core` | `partial` | `crates/elroy-core/src/lib.rs` | Session/turn boundaries exist, the orchestrator accumulates both stream events and normalized transcript messages, and a live provider adapter can drive real provider calls; transcript repair for broken assistant/tool message sequences now exists, but higher-level repository composition and recall workflow parity are still missing |
| `elroy/llm/` | model client, parsing, stream handling | `crates/elroy-llm` | `partial` | `crates/elroy-llm/src/lib.rs` | Provider-neutral stream events, tool-call accumulation, validated context-message modeling, request builders, and live HTTP clients for OpenAI/Anthropic exist; streaming and full runtime integration are still missing |
| `elroy/tools/` | tool schema, registration, execution | `crates/elroy-tools` and `crates/elroy-app` | `partial` | `crates/elroy-tools/src/lib.rs`, `crates/elroy-app/src/lib.rs` | Canonical schema plus OpenAI/Anthropic adapters exist, and the shared runtime now has executable tools for memory/agenda listing, exact detail lookup, file-backed memory updates/archive, and agenda update/complete/delete flows; broader tool coverage and richer write workflows are still missing |
| `elroy/db/` | DB manager, sessions, migrations | `crates/elroy-db` | `partial` | `crates/elroy-db/src/lib.rs` | SQLite connection, migrations, persisted bootstrap inventory, and derived memory/agenda tables rebuilt from files exist; full domain schema and higher-level repository workflows are not implemented |
| `elroy/repository/context_messages/` | context read/refresh workflow | `crates/elroy-db`, `crates/elroy-core`, and `crates/elroy-app` | `partial` | `crates/elroy-db/src/lib.rs`, `crates/elroy-core/src/lib.rs`, `crates/elroy-app/src/lib.rs` | SQLite-backed context-message persistence and transcript continuation exist, the shared runtime now reloads/persists transcript state for both CLI prompt execution and TUI submission, and transcript loading now repairs orphaned or incomplete assistant/tool-call sequences before prompt execution; repository/session abstractions and full workflow parity are still missing |
| `elroy/repository/memories/` | memory lifecycle, recall, consolidation | `crates/elroy-memory` | `partial` | `crates/elroy-memory/src/lib.rs`, `crates/elroy-app/src/lib.rs` | File-backed memory creation, update, archive, and filename/path handling now live in a dedicated crate and are exercised through the shared runtime; prompt-time memory recall now injects transient synthetic `search_memories` context using conservative greeting/ack skip heuristics, recent transcript context, dedupe against transcript-visible recalled memories, and token-overlap matching over active memories, but classifier parity, embeddings-backed recall quality, consolidation, and richer lifecycle workflows are still missing |
| `elroy/repository/reminders/` | reminders and due-item logic | `crates/elroy-app`, `crates/elroy-db`, and `crates/elroy-agenda` | `partial` | `crates/elroy-app/src/lib.rs`, `crates/elroy-db/src/lib.rs`, `crates/elroy-agenda/src/lib.rs` | Active due-item listing plus file-backed due-item create/update/rename/complete/delete flows now exist on top of agenda-backed trigger fields, and active due items are now surfaced into prompt context as transient synthetic tool context before model execution; dedicated reminder-domain crate and richer recall surfacing are still missing |
| `elroy/repository/agenda/` | agenda item workflows | `crates/elroy-agenda` | `partial` | `crates/elroy-agenda/src/lib.rs`, `crates/elroy-app/src/lib.rs`, `crates/elroy-db/src/lib.rs` | File-backed agenda creation, update-log appends, completion, deletion, checklist add/edit/complete flows, and an agenda-only query view that excludes reminder-triggered items now exist; richer edits and broader workflow parity are still missing |
| `elroy/repository/tasks/` | task mutation workflows | `crates/elroy-tasks` and `crates/elroy-app` | `partial` | `crates/elroy-tasks/src/lib.rs`, `crates/elroy-app/src/lib.rs` | A dedicated task crate now wraps unified agenda-backed task file mutations and task-oriented queries for active/triggered/due/today task views, and the live runtime now exposes task create/show/list/update/rename/complete/delete tools with optional date and trigger metadata on creation; recall-index and context-bridge side effects from the Python orchestrator are still missing |
| `elroy/repository/user/` | user session and preferences | `crates/elroy-user`, `crates/elroy-db`, and `crates/elroy-app` | `partial` | `crates/elroy-user/src/lib.rs`, `crates/elroy-db/src/lib.rs`, `crates/elroy-app/src/lib.rs` | SQLite-backed persisted assistant name, preferred name, full name, and persona now exist for the local user; live tools can set/get names, update/reset persona, and prompt execution now renders the Python persona template with persisted assistant/user naming. Broader session identity behavior is still missing |
| `elroy/repository/codex_sessions/` | Codex session persistence and workflows | `crates/elroy-codex`, `crates/elroy-db`, and `crates/elroy-app` | `partial` | `crates/elroy-codex/src/lib.rs`, `crates/elroy-app/src/lib.rs` | SQLite-backed Codex session persistence now exists with per-user upsert/get/list behavior, async dispatch/resume flows that manage isolated git worktrees and merge into the long-lived `agent` branch, plus live tools to launch, resume, list, and inspect sessions; session-file discovery, background completion follow-up messaging, and TUI/sidebar integration are still missing |
| `elroy/messenger/` | stream/event messaging loop | `crates/elroy-core` or `crates/elroy-messenger` | `not started` | none | May merge into conversation orchestration if boundaries stay clear |
| `elroy/ui/` | Textual TUI | `crates/elroy-tui` | `partial` | `crates/elroy-tui/src/lib.rs`, `crates/elroy-cli/src/main.rs`, `crates/elroy-app/src/lib.rs` | `ratatui` shell, focus/key-mode logic, and a runnable terminal loop exist; the TUI now submits prompts through the shared runtime, can open DB-backed memory/agenda detail into the conversation pane, and supports initial sidebar mutation hotkeys for archive/complete/delete, but richer session and mutation flows are still missing |
| `tests/` | Python behavioral coverage | Rust test suites | `not started` | none | Reuse fixtures and scenarios where practical |

## Cross-Cutting Parity Requirements

These are not single modules, but they must be tracked during implementation:

| Requirement | Status | Tests | Notes |
| --- | --- | --- | --- |
| Config precedence and defaults | `partial` | `crates/elroy-config/src/lib.rs` | Default config values, YAML loading, and env precedence exist; most config keys remain unimplemented |
| Tool schema compatibility | `partial` | `crates/elroy-tools/src/lib.rs`, `crates/elroy-app/src/lib.rs` | Canonical schema and provider adapters exist, and a growing set of executable memory/agenda tools now run through the shared live runtime; broader schema coverage and richer execution semantics remain unimplemented |
| Streaming status semantics | `partial` | `crates/elroy-llm/src/lib.rs`, `crates/elroy-core/src/lib.rs` | Event types exist, the core loop consumes them, normalized transcript messages are built alongside events, and non-streaming provider response parsing exists; true streaming integration is still missing |
| TUI keyboard behavior | `partial` | `crates/elroy-tui/src/lib.rs` | Source of truth is `../elroy/UI.md`; focus toggling, pane switching, and key intents are modeled and bound to terminal events, and submit/open/archive/complete/delete actions now call a runtime interface, but broader session workflows are still missing |
| Persistence compatibility with existing user data | `partial` | `crates/elroy-db/src/lib.rs`, `crates/elroy-cli/src/main.rs` | Compatible file discovery plus rebuild of derived bootstrap, memory, agenda, and local context-message state exist; higher-level repository behavior is still missing |
| File-backed inspectable recallable content | `partial` | `crates/elroy-db/src/lib.rs`, `crates/elroy-app/src/lib.rs` | Recursive file inventory, frontmatter/body extraction, derived memory/agenda table rebuilds, and a first set of file-backed memory/agenda mutation helpers exist for markdown-backed data; repository-level sync behavior is not implemented yet |
| Memory recall quality | `partial` | `crates/elroy-app/src/lib.rs` | Prompt-time recall now exists via deterministic heuristic matching over active memories with recent transcript context and transcript-level dedupe, but scenario-driven parity tests and embeddings/classifier behavior are still missing |
| Reminder and due-item surfacing | `partial` | `crates/elroy-app/src/lib.rs`, `crates/elroy-core/src/lib.rs` | Active due items are now injected into model context through transient synthetic tool messages, but timing-sensitive selection and richer contextual surfacing are still missing |
| Codex workflow support in v1 | `partial` | `crates/elroy-codex/src/lib.rs`, `crates/elroy-app/src/lib.rs` | Persisted session metadata, async dispatch/resume lifecycle, isolated worktree handling, and tool-surface inspection now exist, but session-file discovery, assistant follow-up behavior, and UI workflow support are still missing |

## Intentional Deltas

Record accepted behavior changes here once they exist.

| Area | Delta | Rationale | Approved by | Date |
| --- | --- | --- | --- | --- |
| none yet | none | none | none | none |
