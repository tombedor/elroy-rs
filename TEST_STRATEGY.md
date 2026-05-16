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

- `elroy-config`: defaults, YAML parsing, unknown-key tolerance, env override precedence, greeting/bootstrap context controls, and context-refresh token controls
- `elroy-core`: session and turn boundaries plus provider-neutral conversation/tool-loop orchestration with normalized transcript accumulation, lazy streamed tool-loop support, and live-provider runtime wiring
- `elroy-db`: bootstrap planning, recursive markdown discovery including separate archived-memory discovery, migration execution, frontmatter parsing, persisted bootstrap inventory, derived memory/agenda table rebuilds including inactive archived-memory history rows, and context-message persistence
- `elroy-memory`: file-backed memory filename handling plus create/update/archive operations
- `elroy-agenda`: file-backed agenda create/update/complete/delete operations plus checklist add/edit/complete behavior, including returned update timestamps
- `elroy-feature-requests`: markdown-backed feature-request create/load/list/update flows, duplicate matching, and active self-reflection filtering
- `elroy-self-reflection`: correction-triggered proposal generation, cadence gating, dedupe/reopen behavior, and disabled-zero semantics
- `elroy-db`: memory-operation-tracker migration and round-trip persistence
- `elroy-db`: agenda-only vs due-item query separation plus derived checklist counts
- `elroy-db`: context-message-set get-or-create stability for one active set per user
- `elroy-app` + `elroy-db` + `elroy-agenda`: active/inactive due-item query behavior, Python-style timed-due prompt surfacing including multiple simultaneous due items plus app-level prompt/runtime surfacing, contextual-trigger surfacing including persisted current-context pinning and later-turn dedupe for matched reminders, hybrid timed-reminder surfacing, handling and cleanup through `delete_due_item`, app-level omission of future timed reminders from the prompt transcript, Python-named active due-item inspection plus active/inactive listing aliases, and file-backed due-item create/update/rename/complete/delete flows including delete-then-recreate same-name reminder reuse
- `elroy-app` + `elroy-db`: current-context memory pinning/drop behavior including same-name cross-store scoping for `add_memory_to_current_context`, exact-name memory read/mutate/source tool scoping to the active configured `memory_dir`, same-name cross-store list/search/examine memory-tool scoping, prompt-time memory-recall scoping to the active configured `memory_dir`, exact-duplicate consolidation scoping to the active configured `memory_dir`, and archived-memory derived-row history for outdated-memory replacement
- `elroy-llm`: stream event model, partial tool-call accumulation, validated context-message rules, provider request payload building including transcript-system-message dedupe for live system prompts, live HTTP client request shaping, non-streaming response parsing, and SSE-backed OpenAI/Anthropic stream parsing
- `elroy-app`: shared runtime behavior for provider config translation, snapshot loading, session-open stale-context pruning, prompt execution, runtime context-load repair of missing/misplaced system messages plus Anthropic-style synthetic first-user insertion, Python-style context-refresh token-budget/compression helpers plus deferred refresh execution and synthetic `context_summary` persistence, startup restart/greeting stream handling, Python-style prompt prelude status ordering (`loading context...`, recall status, `thinking...`), Python-style timed-due reminder surfacing with cleanup guidance, the `⏰ DUE ITEM:` prefix, normalized schedule formatting, plus heuristic contextual-trigger surfacing that now persists the matched reminder into current context and avoids duplicating it on later matching turns, prompt-time memory recall classifier/window controls including case-insensitive trivial greeting/acknowledgment skip coverage, `search_memories` due-item plus plain-agenda-item inclusion with a Python-closer user-facing search report, explicit memory pin/drop tool behavior for persisted transcript context plus `get_fast_recall` acknowledgement, formatted `examine_memories` output over matching memories, due items, and plain agenda items with Python-style source-drilldown guidance, a first `get_source_list_for_memory` surface over persisted source metadata that now returns the Python-style bare source list, and a broader `get_source_content_for_memory` path that can now return persisted context-message source content for transcript-derived memories plus archived prior-memory source content for outdated-memory replacements and archived multi-memory source content for consolidated-memory lineage. Memory-backed source content now also includes the archived source memory’s Python-style `#name` fact header instead of only the raw body text. Plain memories without explicit source metadata now return an empty source list and “No sources found…” instead of a raw file fallback, and source-content index errors now use Python-style `Index ... out of range` wording, while `print_memory` now uses the Python `#name` fact-style format and exposes only the Python `memory_name` schema, `print_memories` now lists visible memories oldest-first and exposes only the Python `n` schema field, the active/inactive due-item print-list tools now also expose only the Python `n` schema field, the memory search/examination tools now expose only the Python `query` / `question` schema fields, and the task list tools now expose no Rust-only `limit` parameter, `print_due_item` and the due-item print lists now normalize timed trigger display to Python’s `YYYY-MM-DD HH:MM:SS` format, `create_memory` now exposes only the Python `name`/`text` schema and resets the auto-memory tracker the same way Python’s manual-memory path does, `create_due_item` now exposes only the Python `name`/`text`/`trigger_time`/`trigger_context` schema, `update_due_item_text` now exposes only the Python `name`/`new_text` schema, `rename_due_item` now exposes only the Python `old_name`/`new_name` schema, `rename_task` now exposes only the Python `old_name`/`new_name` schema, source inspection, memory creation confirmation text, due-item print/mutation behavior, task creation context pinning plus refresh/removal across task update/rename/complete/delete, due-item context pinning plus refresh/removal across due-item update/rename/complete/delete, task mutation missing/duplicate-rename behavior, due-item create/delete fast-recall context side effects, agenda/reminder file deletion semantics, and Python-style agenda creation/mutation argument defaults now use direct Python-style wording/side effects, including text-derived naming, default-today dates, the `item_date` alias, Python-style case-insensitive unique-substring agenda lookup plus the matching no-match/ambiguous-match errors on the agenda-item tool path, `item_name` / `checklist_item_id` / `new_text` agenda mutation aliases, direct Python-style missing-checklist-id wording for agenda checklist edit/complete failures, real timestamp propagation in `add_agenda_item_update`, Python-style `new_text` acceptance on `update_due_item_text`, Python-style `item_date` acceptance on `create_task` plus schema coverage for task trigger/date metadata, duplicate-name rejection for task creation and contextual/timed due-item creation, blank-name rejection for task and due-item creation, missing-trigger rejection for due-item creation, past-trigger rejection for task and timed due-item creation, and due-task filtering that excludes future timed plus context-only tasks, `old_name` rename aliases for tasks and due items, `n` aliases on the Python-style memory/reminder print-list surfaces, `memory_name` aliases on the memory detail and direct update/archive tools, and the Python-style non-error missing-memory response for `update_outdated_or_incorrect_memory`. The runtime now also exposes a test-covered `create_consolidated_memory` tool path that archives named source memories and preserves their lineage metadata, the outdated-memory replacement path now preserves an inactive archived derived row in the rebuilt `memories` table, and the memory tracker now has a first exact-duplicate automatic consolidation pass when the configured threshold is crossed. It also covers Python-style base time/filesystem tools (`get_current_date`, `pwd`, bounded `ls`, bounded `read_file`), Python-style developer commands (`get_help`, `print_config`, `tail_elroy_logs`) with structured plain-text table output for help/config reports and the first memory consolidation/recall-quality config rows, Python-named date-scoped `list_agenda_items` plus formatted `list_agenda_items_cmd`, Python-named `restart_session` scheduling when interactive restart support is enabled, Python-closer outdated-memory replacement behavior with source preservation, Python-style `trigger_time` alias plus confirmation strings for due-item creation, Python-style human confirmation text for due-item update/rename/complete/delete, Python-style human confirmation text for agenda add/update/complete/delete and checklist mutations, task mutation confirmation text plus persisted complete/delete closing comments, Python-named print/report formatting for memory and reminder user-facing commands, Python-named memory show/list aliases, archive-memory rebuild semantics that keep `memory_dir/archive` entries out of the active memory surface while retaining inactive derived history rows, post-persist self-reflection triggering, message-count-driven auto-memory creation, Python-named context reset/refresh tool entries with persisted leading system-message behavior, DB-backed sidebar-detail loading, Python-style agenda-sidebar title formatting and title-to-item resolution, file-backed feature-request sidebar reads/close mutations, Python-style feature-request list/create/edit tool flows with duplicate-aware merge behavior, and file-backed memory/agenda mutation tools
- `elroy-core`: shared turn orchestration, validated transcript repair including wrong-tool-call-id dropping and multi-tool-call retention, force-tool forwarding, Python-style tool-loop status ordering around local tool execution in both buffered and streamed paths, and the shared background-status registry
- `elroy-tools`: canonical tool schema, provider adapter projections, and executable local registry behavior including DB-backed read/write tools in the shared runtime
- `elroy-tui`: layout shell, focus-mode state machine, terminal key mapping, event-loop action transitions, runtime-backed prompt submission, startup restart/greeting stream handling, TUI-local `restart_session` follow-up streaming, Python-style rendered-message bookkeeping after bootstrap/chat turns, deferred context-refresh scheduling, background context polling/rendering rules, live prompt-stream consumption with blocked resubmit and `Ctrl+C` clear/cancel behavior, idle footer rendering for model/background status, runtime-backed sidebar section switching/opening, detail-modal confirmation behavior for destructive sidebar actions, runtime-backed sidebar mutation actions including feature-request completion, and snapshot rendering
- `elroy-codex` + `elroy-app`: persisted Codex-session upsert/get/list behavior, async dispatch/resume workflow orchestration with isolated git worktrees, background-status registration across running sessions, and live session launch/resume/inspection tools
- `elroy-tui` + `elroy-app`: read-only Codex-session sidebar rendering, section switching, and detail opening through the shared runtime
- Direct Python-scenario ports now exist for selected messaging/Codex/UI behaviors, including `persist_input_message=False`, force-tool request plumbing, incremental provider-stream parsing and streamed tool-loop orchestration, startup restart/greeting stream handling, runtime session-open stale-context pruning via `max_context_age_minutes`, persisted leading-system-message behavior for Python-named context reset/refresh and user-preference updates, runtime repair of missing/misplaced system messages before prompt execution and explicit context loads, Anthropic-style synthetic first-user insertion plus snapshot suppression of that line, Python-style context-refresh token-threshold detection, cache-friendly compression ordering/tool-pair preservation, deferred refresh execution, and synthetic `context_summary` persistence, recall-classifier config gating/window behavior, rendered-message bookkeeping after bootstrap/chat turns, background context polling/rendering rules, agenda-sidebar due-label/title-resolution behavior, feature-request duplicate matching plus `list` / `make` / `edit` tool behavior, self-reflection cadence/correction/reopen behavior, message-count-driven auto-memory creation behavior, Python-named context reset tool behavior, feature-request/improvement sidebar visibility and completion behavior, detail-modal confirmation behavior for sidebar actions, TUI draft-editability/cancel behavior during prompt streams, Codex background-status registration and completion persistence, Codex resume/list flows, and Codex sidebar visibility

This is only bootstrap-level coverage. It does not yet prove product parity.
