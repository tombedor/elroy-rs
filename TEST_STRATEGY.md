# Test Strategy

This document defines how `elroy-rs` should prove parity with `../elroy`.

The rewrite should rely on tests as the primary proof of correctness. Architecture and code review matter, but they are not substitutes for executable parity checks.

## Principles

- test user-visible behavior, not just implementation details
- document what each test layer proves
- prefer small deterministic fixtures over broad implicit setup
- use Python behavior and fixtures as the baseline where practical
- record known coverage gaps in `PARITY_MATRIX.md`

## Test Layers

### Unit Tests

Use for:

- config parsing and precedence
- path/layout derivation
- prompt/builder logic
- pure decision logic
- parser behavior

These should be fast, deterministic, and cheap to run continuously.

### Store / Persistence Tests

Use for:

- file-backed storage behavior
- migration logic
- schema and query behavior
- rebuild/bootstrap of derived DB state from file inputs

These should use realistic temporary file trees and database fixtures.

### Orchestrator Tests

Use for:

- turn processing workflows
- tool-call execution ordering
- context loading and persistence
- recall and due-item injection rules

These should validate workflow semantics without requiring the full TUI.

### Golden / Snapshot Tests

Use for:

- tool schema output
- streaming event ordering
- formatted prompt/context payloads
- stable CLI-visible reports where format matters

These should only be used where stable textual structure is part of the contract.

### End-To-End Tests

Use for:

- CLI entry behavior
- TUI interaction flows
- bootstrap from a compatible Elroy home directory
- Codex workflow integration once implemented

These should cover the default workflows users actually rely on.

## Required Coverage By Area

| Area | Minimum expectation |
| --- | --- |
| Config | unit tests for defaults, file loading, env precedence, compatibility keys |
| File-backed memory and agenda data | persistence tests using realistic file trees |
| DB bootstrap/rebuild | persistence or integration tests starting from compatible files |
| Tool system | golden tests for schemas plus orchestrator tests for execution behavior |
| Conversation loop | orchestrator tests for event ordering and persistence effects |
| TUI | end-to-end or UI integration tests for keybindings and focus behavior |
| Codex workflow | integration tests for persistence and lifecycle behavior |

## Documentation Rule

Each subsystem should make it easy to answer:

- what Python behavior is this test proving?
- what compatibility guarantee does this suite cover?
- what is still intentionally untested?

## Current Baseline

Current implemented coverage:

- `elroy-config`: defaults, YAML parsing, unknown-key tolerance, env override precedence, and greeting-bootstrap controls
- `elroy-core`: session and turn boundaries plus provider-neutral conversation/tool-loop orchestration with normalized transcript accumulation, lazy streamed tool-loop support, and live-provider runtime wiring
- `elroy-db`: bootstrap planning, recursive markdown discovery, migration execution, frontmatter parsing, persisted bootstrap inventory, derived memory/agenda table rebuilds, and context-message persistence
- `elroy-memory`: file-backed memory filename handling plus create/update/archive operations
- `elroy-agenda`: file-backed agenda create/update/complete/delete operations plus checklist add/edit/complete behavior
- `elroy-feature-requests`: markdown-backed feature-request create/load/list/update flows, duplicate matching, and active self-reflection filtering
- `elroy-self-reflection`: correction-triggered proposal generation, cadence gating, dedupe/reopen behavior, and disabled-zero semantics
- `elroy-db`: agenda-only vs due-item query separation plus derived checklist counts
- `elroy-app` + `elroy-db` + `elroy-agenda`: active due-item query behavior plus file-backed due-item create/update/rename/complete/delete flows
- `elroy-llm`: stream event model, partial tool-call accumulation, validated context-message rules, provider request payload building, live HTTP client request shaping, non-streaming response parsing, and SSE-backed OpenAI/Anthropic stream parsing
- `elroy-app`: shared runtime behavior for provider config translation, snapshot loading, prompt execution, startup restart/greeting stream handling, Python-style prompt prelude status ordering (`loading context...`, recall status, `thinking...`), prompt-time memory recall classifier/window controls, post-persist self-reflection triggering, DB-backed sidebar-detail loading, Python-style agenda-sidebar title formatting and title-to-item resolution, file-backed feature-request sidebar reads/close mutations, Python-style feature-request list/create/edit tool flows with duplicate-aware merge behavior, and file-backed memory/agenda mutation tools
- `elroy-core`: shared turn orchestration, validated transcript repair, force-tool forwarding, Python-style tool-loop status ordering around local tool execution in both buffered and streamed paths, and the shared background-status registry
- `elroy-tools`: canonical tool schema, provider adapter projections, and executable local registry behavior including DB-backed read/write tools in the shared runtime
- `elroy-tui`: layout shell, focus-mode state machine, terminal key mapping, event-loop action transitions, runtime-backed prompt submission, startup restart/greeting stream handling, Python-style rendered-message bookkeeping after bootstrap/chat turns, background context polling/rendering rules, live prompt-stream consumption with blocked resubmit and `Ctrl+C` clear/cancel behavior, idle footer rendering for model/background status, runtime-backed sidebar section switching/opening, detail-modal confirmation behavior for destructive sidebar actions, runtime-backed sidebar mutation actions including feature-request completion, and snapshot rendering
- `elroy-codex` + `elroy-app`: persisted Codex-session upsert/get/list behavior, async dispatch/resume workflow orchestration with isolated git worktrees, and live session launch/resume/inspection tools
- `elroy-tui` + `elroy-app`: read-only Codex-session sidebar rendering, section switching, and detail opening through the shared runtime
- Direct Python-scenario ports now exist for selected messaging/Codex/UI behaviors, including `persist_input_message=False`, force-tool request plumbing, incremental provider-stream parsing and streamed tool-loop orchestration, startup restart/greeting stream handling, recall-classifier config gating/window behavior, rendered-message bookkeeping after bootstrap/chat turns, background context polling/rendering rules, agenda-sidebar due-label/title-resolution behavior, feature-request duplicate matching plus `list` / `make` / `edit` tool behavior, self-reflection cadence/correction/reopen behavior, feature-request/improvement sidebar visibility and completion behavior, detail-modal confirmation behavior for sidebar actions, TUI draft-editability/cancel behavior during prompt streams, Codex background completion persistence, Codex resume/list flows, and Codex sidebar visibility

This is only bootstrap-level coverage. It does not yet prove product parity.
