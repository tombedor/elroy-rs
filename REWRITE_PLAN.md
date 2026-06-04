# Rewrite Roadmap

This document is the execution roadmap for the remaining rewrite work in `elroy-rs`.

It supersedes the earlier bootstrap-oriented milestone list. The workspace now has substantial partial parity across most major subsystems, so the next planning problem is sequencing the remaining work into usable product checkpoints instead of continuing an undifferentiated stream of small parity fixes.

Source of truth for status remains [PARITY_MATRIX.md](/Users/tombedor/development/elroy-rs/PARITY_MATRIX.md). This document answers a different question:

- what should we do next
- in what order
- what can wait
- what counts as a usable checkpoint along the way

## Planning Principles

1. Ship usable vertical checkpoints, not endless micro-parity patches.
2. Prefer closing broad `partial` rows that affect core workflows over polishing already-usable tool wording or schema details.
3. Only do narrow contract or wording fixes when they:
   - unblock a phase exit criterion
   - close a high-frequency user-facing mismatch in an otherwise finished flow
   - protect existing data or workflow correctness
4. Keep Rust behavior aligned with Python unless an intentional delta is documented in the parity matrix.
5. Every phase should leave the product in a better standalone state, not just a more complete matrix.
6. Tests accompany new behavior; coverage-only commits are Phase 6 work. Writing a test for already-implemented behavior does not advance the current phase. A commit that only adds tests — with no behavioral change — belongs in Phase 6 regardless of which phase you are in. During Phases 1–5, write tests when they verify a behavior change in the same commit. If several consecutive commits are coverage-only, treat that as a signal to stop and redirect toward the phase exit criteria.

## Current Read Of The Product

The rewrite is no longer in a "missing major subsystems everywhere" state.

Implemented enough to be structurally usable:

- shared runtime and messenger loop
- live provider streaming
- file-backed memories, tasks, reminders, agenda items
- current-context pinning flows
- self-reflection and feature-request slices
- Codex session persistence and background follow-up
- a working ratatui shell with prompt streaming, sidebars, modals, and background polling

What is still broadly incomplete is concentrated in a smaller set of high-value areas:

- memory recall quality
- context refresh quality and orchestration
- broader TUI/session workflow parity
- richer Codex interactive UI workflows
- broader config/tool coverage
- broader repository-level persistence behavior
- more representative end-to-end parity coverage

Recent Phase 2 progress has closed two concrete startup/session gaps: the Rust CLI/TUI path now mirrors Python’s hidden `get_session_context` bootstrap before greeting or restart evaluation, while keeping that synthetic bootstrap pair out of the visible transcript, and prompt history now persists across TUI restarts with Python-style exclusion of slash commands from recalled history. The older “multi-session history/switching” wording was too broad; the Python UI source does not expose a separate session-switching workflow, so the remaining Phase 2 TUI work is now better described as runtime/UI coordination and background-rendering edge cases.

That means sequencing should now optimize for "usable release candidate" behavior, not for raw parity-matrix row count.

## What Not To Prioritize

Do not spend primary effort on these until the current phase says they matter:

- one-off tool wording tweaks
- additional schema narrowing for already-usable tools
- isolated print/report formatting differences
- extra direct parity tests for behavior that is already well-covered through larger flows
- coverage-only commits for already-working behavior while Phase 2 exit criteria remain unmet — this is the single most common way to consume velocity without advancing the product

Those are acceptable as opportunistic cleanup inside a broader slice, but they should not drive roadmap order.

## Roadmap Phases

## Phase 1: Structural Refactoring

Goal:

- break `elroy-app` (currently ~28K LOC, ~60% of the codebase) into well-bounded crates that match the `ARCHITECTURE.md` target shape
- no user-visible behavior changes; this phase is purely structural
- make it possible to evolve recall, context refresh, and tool execution independently

Why this comes first:

- `elroy-app` has grown into a monolith that absorbs tool execution, recall orchestration, consolidation, context refresh, reminder surfacing, and formatting — all in one 28K-line file
- the 14 other crates are mostly storage-only stubs; their intended orchestration logic ended up in `elroy-app`
- this makes Phase 2 and 3 improvements (recall quality, context refresh quality) expensive to implement because all the logic is co-located with unrelated concerns
- fixing structure now makes every subsequent phase faster and better tested

### Session Progress (2025-05-18)

**Done:**

- `elroy-recall` crate created and fully populated (~1,800 LOC moved). All recall structs, classification, selection, embedding, reflective recall logic moved from `elroy-app`. `elroy-app/src/recall.rs` is now a 2-line re-export. Committed.

- `elroy-context` crate created and fully populated (~580 LOC moved). All transcript loading/validation, system message building, context compression/summary logic moved from `elroy-app`. `elroy-app/src/context.rs` is now a 2-line re-export. Committed.

- `LOCAL_USER_TOKEN` and `SYNTHETIC_FIRST_USER_MESSAGE` constants moved to `elroy-db` as `pub const`, eliminating the duplicated string literals scattered across crates.

- Provider config conversion functions (`provider_config_from_app_config`, `fast_provider_config_from_app_config`, `embedding_provider_config_from_app_config`) moved from `elroy-app` to `elroy-config` (added `elroy-llm` dep to `elroy-config`). Eliminates 80-line duplication in `elroy-app`.

- `From<anyhow::Error> for AppError` added to `elroy-app` so domain crates returning `anyhow::Result` can interoperate with `elroy-app`'s error type via `?`.

### Session Progress (2026-05-24)

**Done:**

- `crates/elroy-agenda/src/tools.rs` now owns the agenda and due-item tool execution slice, and `crates/elroy-agenda/src/lib.rs` exports `agenda_tools(...)` for the app registry.

- `crates/elroy-tools/src/base.rs` now owns the base filesystem/developer tool slice (`get_current_date`, `pwd`, `ls`, `read_file`, `restart_session`, `print_config`, `tail_elroy_logs`, `get_help`), and `elroy-app` now only provides the restart/help/config-report callbacks needed to compose those tools into the live registry.

- `crates/elroy-context/src/tools.rs` now owns the persisted transcript management slice (`show_context_messages`, `reset_messages`, `refresh_system_instructions`), and `crates/elroy-memory/src/tools.rs` now owns the recall-dependent memory pin/drop slice (`add_memory_to_current_context`, `drop_memory_from_current_context`). `elroy-app` no longer has a dedicated context-tools module.

- `crates/elroy-reminders/src/lib.rs` now owns the due-item surfacing wrapper layer used by prompt execution: timed due-item synthetic context, contextual due-item context-message assembly, and the shared orchestration that merges timed/contextual reminder surfacing before a model turn.

- The other domain tool slices are already extracted and wired through the live registry: `elroy-memory/src/tools.rs`, `elroy-tasks/src/tools.rs`, `elroy-user/src/tools.rs`, `elroy-feature-requests/src/tools.rs`, and `elroy-codex/src/tools.rs`.

- The extracted due-item tool paths were brought back to current parity expectations for this slice: contextual due-item creation re-pins context, due-item completion/deletion preserve the Python-style confirmation text and context cleanup, and recreating a completed due item now reclaims the canonical logical name so follow-up lookup/context flows still work.

**Validation run for this slice:**

- `cargo fmt --all -- crates/elroy-agenda/src/lib.rs crates/elroy-agenda/src/tools.rs crates/elroy-tools/src/lib.rs crates/elroy-tools/src/base.rs crates/elroy-context/src/lib.rs crates/elroy-context/src/tools.rs crates/elroy-memory/src/tools.rs crates/elroy-reminders/src/lib.rs crates/elroy-recall/src/lib.rs crates/elroy-app/src/lib.rs`
- `cargo check -p elroy-app -p elroy-tools -p elroy-context -p elroy-memory`
- `cargo test -p elroy-tools -- --nocapture`
- `cargo test -p elroy-codex -- --nocapture`
- `cargo test -p elroy-agenda -- --nocapture`
- `cargo test -p elroy-feature-requests -- --nocapture`
- `cargo test -p elroy-tasks -- --nocapture`
- `cargo test -p elroy-user -- --nocapture`
- `cargo test -p elroy-memory -- --nocapture`
- `cargo test -p elroy-context -- --nocapture`
- `cargo test -p elroy-reminders -- --nocapture`
- `cargo test -p elroy-app live_tool_registry_can_`
- `cargo clippy -p elroy-app -p elroy-agenda -p elroy-codex -p elroy-tools -p elroy-context -p elroy-reminders -p elroy-recall -p elroy-tasks -p elroy-user --all-targets --all-features`
- `cargo test -p elroy-app`
- `just lint`
- `just test`
- `cargo build --workspace`

**State after session:**

- `elroy-app/src/lib.rs`: 1,309 LOC total after the base-tool, context-tool, reminder-surfacing, additional crate-local test moves, the internal registry/restart extraction, the TUI helper extraction, the runtime-helper extraction, and the final app-test extraction into `crates/elroy-app/src/tests.rs`
- `crates/elroy-app/src/tests.rs`: 15,487 LOC of app-owned integration and runtime coverage that no longer bloats the app boundary file itself
- Consolidation helpers now live in `elroy-recall/src/lib.rs`; the old `elroy-app/src/consolidation.rs` note is obsolete
- The remaining structural extraction gap inside the live registry is now narrower than the older plan implied: the major tool families already live in their destination crates, and `elroy-app` is mostly a combiner plus the broader runtime/TUI-facing orchestration layer rather than an owner of individual tool implementations
- `elroy-context/src/tools.rs` now also owns the direct reset/refresh/show behavior coverage for its extracted transcript tools instead of leaving that slice under `elroy-app`'s live-registry tests
- `elroy-tools/src/base.rs` now also owns the direct filesystem/time/help/restart/log-tail behavior coverage for its extracted base tools instead of leaving those cases under `elroy-app`'s live-registry tests
- `elroy-feature-requests/src/tools.rs` now also owns the direct list/create/merge/edit behavior coverage for its extracted markdown feature-request tools instead of leaving that slice under `elroy-app`'s live-registry tests
- `elroy-user/src/tools.rs` now also owns the direct persisted-preferences and refreshed-system-message behavior coverage for its extracted user-preference tools instead of leaving that slice under `elroy-app`'s live-registry tests
- `elroy-tasks/src/tools.rs` now also owns the direct create/update/rename/complete/delete/list behavior coverage for its extracted task tools, including context-refresh effects and due-task filtering, instead of leaving that slice under `elroy-app`'s live-registry tests
- `elroy-codex/src/tools.rs` now also owns the direct list/show session behavior coverage for its extracted codex tools instead of leaving that slice under `elroy-app`'s live-registry tests; the background dispatch/resume workflow test still remains in `elroy-app` because it exercises the app-owned completion-hook seam
- `elroy-agenda/src/tools.rs` now also owns the direct checklist-item behavior coverage for its extracted agenda tools instead of leaving that slice under `elroy-app`'s live-registry tests
- `elroy-agenda/src/tools.rs` now also owns the direct inactive due-item listing/detail behavior coverage for its extracted due-item tools instead of leaving that slice under `elroy-app`'s live-registry tests
- `elroy-agenda/src/tools.rs` now also owns the direct agenda-item mutation coverage and the direct due-item create/update/rename/complete/delete behavior coverage for its extracted agenda/due-item tools instead of leaving those slices under `elroy-app`'s live-registry tests
- `elroy-agenda/src/tools.rs` now also owns the direct show/list/print agenda-item read-path coverage for its extracted agenda tools instead of leaving those slices under `elroy-app`'s live-registry tests
- `elroy-memory/src/tools.rs` now also owns the direct show/print/list/source-content/source-list behavior coverage for its extracted read-only memory tools instead of leaving that slice under `elroy-app`'s live-registry tests
- `elroy-memory/src/tools.rs` now also owns the direct add/drop pinned-context, update/archive, and outdated-memory-update behavior coverage for its extracted mutation-heavy memory tools instead of leaving those slices under `elroy-app`'s live-registry tests
- `crates/elroy-reminders/src/lib.rs` now owns the contextual due-item selector itself, including overlap/semantic/embedding reminder selection behavior and crate-local coverage for that selector, instead of delegating that reminder-specific seam back into `elroy-recall`
- `crates/elroy-app/src/tool_registry.rs` now owns the live tool-registry composition block plus session-restart support state/callback wiring, reducing `elroy-app/src/lib.rs` production bulk without changing app-owned behavior
- `crates/elroy-app/src/ui_helpers.rs` now owns command-form ordering, snapshot/sidebar formatting, and related TUI-facing helper logic, reducing `elroy-app/src/lib.rs` production bulk without changing app-owned behavior
- `crates/elroy-app/src/runtime_helpers.rs` now owns prompt-finalization, provider-model construction, context-refresh, self-reflection, and background codex follow-up helper logic, reducing `elroy-app/src/lib.rs` production bulk without changing app-owned behavior
- `crates/elroy-agenda/src/tools.rs` now also owns the remaining due-item schema-surface assertions that had been left behind in the app crate, so `elroy-app` no longer carries direct schema checks for that tool family
- Phase 1 structural extraction work is complete; the next remaining product work starts in Phase 2 rather than another round of app-boundary slicing

### Note on 8K LOC Target

The app boundary file target is now satisfied: `crates/elroy-app/src/lib.rs` is 1,309 LOC. App-owned integration tests remain in `crates/elroy-app/src/tests.rs`, while direct tool-behavior coverage has been moved alongside the owning crates where appropriate.

### New Crates To Create

**`elroy-recall`** ✅ Created

Owns all memory recall and consolidation logic currently embedded in `elroy-app`:

- recall classification (heuristic and model-backed) ✅ moved
- candidate selection and expansion (lexical overlap, embedding ranking) ✅ moved
- relevance filtering (model-backed) ✅ moved
- reflective recall generation (deterministic and model-backed) ✅ moved
- exact-duplicate and semantic-cluster consolidation ✅ moved
- embedding cache management ✅ moved
- auto-memory creation from context ✅ moved

`elroy-memory` becomes a pure file I/O + frontmatter Store. All orchestration moves to `elroy-recall`.

**`elroy-context`** ✅ Created

Owns all context message loading and refresh logic currently embedded in `elroy-app`:

- transcript loading and validation (role alternation repair, orphaned tool call repair) ✅ moved
- system message building and repair ✅ moved
- context compression and summary generation (deterministic and model-backed) ✅ moved
- context refresh scheduling and orchestration ✅ moved
- due-item and reminder pinning into transcript context ✅ moved

**`elroy-reminders`** `partial`

Owns due-item surfacing workflows currently embedded in `elroy-app`:

- due-item context message generation ✅ moved
- synthetic tool message creation for surfaced reminders ✅ moved
- prompt-time orchestration that merges timed and contextual reminder surfacing ✅ moved
- reminder selection heuristics ✅ moved
- interplay between due items, tasks, and current context `partial`

The crate is no longer empty, but it is not yet the full home for every due-item/task interaction seam.

### Domain Crates That Expand

Each of these crates gains a `tools` module owning its own tool execution and tool specs. The corresponding match arms and execution functions move out of `elroy-app`.

| Crate | Tools to absorb from elroy-app | Status |
|---|---|---|
| `elroy-memory` | `create_memory`, `show_memory`, `search_memories`, `print_memories`, `update_memory`, `archive_memory`, recall tools | `partial` |
| `elroy-agenda` | `create_agenda_item`, `update_agenda_item`, `complete_agenda_item`, checklist tools, due-item tools | `partial` |
| `elroy-tasks` | `create_task`, `show_task`, `list_tasks`, `update_task`, `complete_task` | `partial` |
| `elroy-user` | `update_user_preferred_name`, `update_assistant_name`, user preference tools | `partial` |
| `elroy-feature-requests` | `create_feature_request`, `list_feature_requests`, `show_feature_request`, `update_feature_request` | `partial` |
| `elroy-codex` | `dispatch_codex_session`, `resume_codex_session`, `list_codex_sessions` | `partial` |

### `elroy-tools` Expands

`elroy-tools` gains base tool implementations currently embedded in `elroy-app`:

- filesystem tools: `ls`, `read_file`
- developer tools: `get_help`, `print_config`, `tail_elroy_logs`, `restart_session`
- context tools: `reset_messages`, `refresh_system_instructions`

Status: `partial`; filesystem/developer tools now live in `crates/elroy-tools/src/base.rs`, while the persisted-context tool slice now lives in `crates/elroy-context/src/tools.rs` and the memory pin/drop tools now live in `crates/elroy-memory/src/tools.rs`

### `elroy-app` After Refactoring

`elroy-app` becomes a thin wiring and routing layer only:

- `AppRuntime` construction and dependency composition
- tool execution routing (dispatching by tool name to domain crate executors)
- snapshot loading (composing from domain crates)
- command palette and command form prefilling
- `process_message` / `load_snapshot` / `load_context_messages` surface API for TUI/CLI

Target size: under 8K LOC (down from ~28K). Current: `crates/elroy-app/src/lib.rs` is 1,309 LOC, with app-owned integration coverage split into `crates/elroy-app/src/tests.rs`.

### Dependency Graph (After)

```
elroy-app
  → elroy-recall          (recall + consolidation orchestration)
  → elroy-context         (context loading + refresh orchestration)
  → elroy-reminders       (due-item surfacing)
  → elroy-{memory,agenda,tasks,user,feature-requests,codex}  (each with tools module)
  → elroy-tools           (base tools)
  → elroy-{db,llm,core,config,self-reflection,codex}
elroy-tui → elroy-app
elroy-cli → elroy-app + elroy-tui
```

No bidirectional dependencies. Domain orchestrators depend on stores; stores do not depend on orchestrators.

### Exit Criteria

- [x] `elroy-recall` created; all recall logic moved from `elroy-app`
- [x] `elroy-context` created; transcript loading, validation, and refresh orchestration moved from `elroy-app`
- [x] Consolidation logic moved from `elroy-app` to `elroy-recall`
- [x] `elroy-reminders` created; due-item surfacing moved from `elroy-app`
- [x] Each domain crate owns its tool execution; match arms removed from `elroy-app`
- [x] `elroy-tools` owns base/filesystem/developer tools
- [x] `elroy-app` is under 8K LOC
- [x] All existing tests pass with no behavioral changes
- [x] `cargo build` and `cargo test` are clean

Usable checkpoint:

- codebase is maintainable enough to make Phase 2 (recall quality) and Phase 3 (context refresh quality) changes without touching unrelated code

### Implementation Guidance

#### Python-to-Rust Module Mapping

Use these mappings to locate the Python logic that corresponds to each new crate. Read the Python source before moving Rust code to understand the intended behavior.

**`elroy-recall`** — read these Python files first:

| Python path | Responsibility |
|---|---|
| `repository/memories/memory_recall_orchestrator.py` | top-level recall entry point; classification → selection → injection |
| `repository/memories/recall_classifier.py` | heuristic + model-backed classification of whether recall is needed |
| `repository/memories/memory_recall_builder.py` | fast recall and reflective recall payload construction |
| `repository/memories/summarizer.py` | reflective recall content synthesis via LLM |
| `repository/memories/consolidation.py` | exact-duplicate and semantic-cluster consolidation workflows |
| `repository/memories/prompts.py` | prompt templates for classification, relevance filtering, reflective recall |
| `repository/memories/background.py` | auto-memory creation triggered after turns |
| `repository/recall/indexer.py` | embedding index maintenance; keeps cached embeddings aligned with active memories |
| `repository/recall/context_bridge.py` | converts recall results into synthetic context messages for injection |
| `repository/recall/queries.py` | DB queries for recall candidate retrieval |
| `repository/memories/queries.py` | memory-specific DB queries (active memories, recently updated, etc.) |

**`elroy-context`** — read these Python files first:

| Python path | Responsibility |
|---|---|
| `repository/context_messages/context_refresh_orchestrator.py` | when and how to trigger context refresh |
| `repository/context_messages/system_prompt_builder.py` | construct system message from config + user persona |
| `repository/context_messages/transforms.py` | context compression: dropping old messages, building summary injections |
| `repository/context_messages/validations.py` | transcript repair: orphaned tool calls, role alternation enforcement |
| `repository/context_messages/factory.py` | context message creation helpers |
| `repository/context_messages/inspect.py` | read-only inspection utilities (token counting, role analysis) |
| `repository/context_messages/store.py` | persistence layer (move to elroy-db queries; don't replicate here) |

**`elroy-reminders`** — read these Python files first:

| Python path | Responsibility |
|---|---|
| `repository/reminders/reminder_orchestrator.py` | due-item surfacing: selection, synthetic message creation, context pinning |
| `repository/reminders/queries.py` | due-item DB queries (by time, by context match) |
| `repository/reminders/factory.py` | due-item creation helpers |

**Domain crate tool modules** — one-to-one Python mapping:

| Python path | Rust target |
|---|---|
| `repository/memories/tools.py` | `crates/elroy-memory/src/tools.rs` |
| `repository/agenda/tools.py` | `crates/elroy-agenda/src/tools.rs` |
| `repository/tasks/task_mutation_orchestrator.py` | `crates/elroy-tasks/src/tools.rs` |
| `repository/user/tools.py` | `crates/elroy-user/src/tools.rs` |
| `repository/feature_requests/tools.py` | `crates/elroy-feature-requests/src/tools.rs` |
| `repository/codex_sessions/tools.py` | `crates/elroy-codex/src/tools.rs` |
| `repository/context_messages/tools.py` | `crates/elroy-tools/src/context_tools.rs` |

#### Tool Execution Routing Pattern

After the refactoring, each domain crate owns its tool execution. `elroy-app` becomes a router only. Do not let `elroy-app` re-accumulate execution logic.

Each domain crate that implements tools exports:

```rust
// In crates/elroy-memory/src/tools.rs (example)
pub fn tool_specs() -> Vec<ToolSpec> { ... }
pub async fn execute(name: &str, args: &JsonValue, ctx: &ToolContext) -> Result<ToolExecutionResult> { ... }
```

`elroy-app` maintains a single dispatch function:

```rust
pub async fn dispatch_tool(name: &str, args: &JsonValue, ctx: &ToolContext) -> Result<ToolExecutionResult> {
    match name {
        n if elroy_memory::tools::owns(n)          => elroy_memory::tools::execute(n, args, ctx).await,
        n if elroy_agenda::tools::owns(n)           => elroy_agenda::tools::execute(n, args, ctx).await,
        n if elroy_tasks::tools::owns(n)            => elroy_tasks::tools::execute(n, args, ctx).await,
        n if elroy_user::tools::owns(n)             => elroy_user::tools::execute(n, args, ctx).await,
        n if elroy_feature_requests::tools::owns(n) => elroy_feature_requests::tools::execute(n, args, ctx).await,
        n if elroy_codex::tools::owns(n)            => elroy_codex::tools::execute(n, args, ctx).await,
        n if elroy_tools::owns(n)                   => elroy_tools::execute(n, args, ctx).await,
        _ => Err(anyhow!("Unknown tool: {name}")),
    }
}
```

Tool specs at startup are composed from each domain crate's `tool_specs()`. The routing table in `dispatch_tool` is the only place in `elroy-app` that knows which crate owns which tool. If a new tool is added to a domain crate, it should not require any change to `elroy-app` other than an entry in this match.

#### Config Injection Pattern

Orchestrators must not receive the full `AppConfig`. Pass only the config slice they need:

```rust
// Wrong — pulls in full config and makes dependencies implicit:
fn new(config: &AppConfig) -> RecallOrchestrator { ... }

// Correct — narrow dependency, explicit contract:
fn classify_recall_needed(msg: &str, ctx: &[ContextMessage], cfg: &RecallConfig) -> bool { ... }
```

Each orchestrator crate should define its own config sub-struct (or re-export the relevant sub-struct from `elroy-config`) and accept that at the call site. `elroy-app` extracts the relevant sub-config and passes it down.

#### Background Work Boundary

`elroy-context` and `elroy-recall` are stateless — they own decision logic and execution logic but not scheduling. The three-layer pattern is:

- **`elroy-context` / `elroy-recall`**: stateless functions; `is_refresh_needed(...)`, `compress(...)`, `select_candidates(...)`
- **`elroy-app`**: calls those functions and decides whether to defer work; owns the deferred-work queue
- **`elroy-cli`**: owns the background worker thread; calls `AppRuntime::poll_deferred_work()` and drives execution

Do not push scheduling concerns into the orchestrator crates.

## Phase 2: Usable Core Product

Goal:

- make the Rust app reliably usable for the main day-to-day local assistant workflow even if some deep parity remains missing

**Daily-driver check:** Before starting a work session, answer this question: "Could I use the Rust TUI as my *only* interface for a full day of Elroy work right now, and would I trust it?" If no, the honest description of *why not* is the Phase 1 priority list. If yes, Phase 1 is done — move to Phase 2.

Why this comes first:

- the product already has most of the core slices
- remaining blockers are mostly around workflow cohesion rather than basic existence
- this phase creates a credible "use Rust by default" checkpoint

Primary parity rows to advance:

- `elroy/ui/`
- `elroy/messenger/`
- `elroy/repository/context_messages/`
- `Streaming status semantics`
- `TUI keyboard behavior`

Remaining gap checklist (derived from parity matrix "still missing" notes; resolve each as implemented or intentional delta before declaring Phase 1 complete):

- [x] Broader session workflows in the TUI — for Phase 1, the already-ported greeting-on-fresh-start and restart/session transitions are sufficient; the remaining gaps are broader multi-session history/switching workflows and belong to the later TUI-focused phases rather than the structural refactor phase
- [x] Deeper command-form validation parity — required fields now validate before final submit at the TUI layer instead of only surfacing a missing-value failure on `Enter`
- [x] Fuller Textual-style command-palette system-command behavior — Python’s surfaced system-command set (`Focus Memories`, `Focus Agenda`, `Refresh System Instructions`, `Reset Messages`) is now present in the Rust palette, with additional Rust-only section focus entries documented as intentional extensions
- [x] Broader background-status producers — the shared Rust footer now covers the real long-running background paths in use (`context-refresh`, `self-reflection`, `auto-memory`, background command execution, Codex dispatch/resume plus completion follow-up); the remaining Python-only worker groups (`session-bootstrap`, `sidebar-refresh`) are foreground or synchronous flows in the current Rust architecture rather than missing shared background producers

Focus areas:

1. Finish the remaining session workflow gaps in the TUI.
   - remaining background-message rendering edge cases
   - stronger foreground/background prompt state coordination
2. Finish broader background-status producer coverage.
   - long-running refresh/reflection/Codex/background operations should all report through the shared footer model
3. Tighten the runtime/UI handoff for live workflows.
   - fewer snapshot-style seams
   - better cancellation and resumed-state behavior
4. Raise end-to-end confidence for the primary interactive loop.
   - prompt -> tool loop -> transcript persistence -> sidebar/background update

Exit criteria:

- the TUI can serve as the default local interface for ordinary chat, reminders, tasks, agenda, and memory usage
- the main background workflow states are visible and understandable in the UI
- restart/resume/session behavior is coherent enough that users are not forced back to Python for ordinary interactive work

Usable checkpoint:

- "Rust as daily-driver local assistant" for single-user interactive use

### Phase 2 Progress (2026-05-24)

- The CLI/TUI startup path now injects the hidden Python-style `get_session_context` bootstrap tool/result pair before greeting or restart evaluation instead of skipping that session bootstrap entirely.
- That bootstrap payload now carries the Python-style local current date/time and first-chat-today greeting hint, and it is filtered back out of visible snapshot/TUI conversation rendering the same way Python hides it from the normal transcript surface.
- Direct coverage now exists in `crates/elroy-app/src/tests.rs` and `crates/elroy-cli/src/main.rs` for both the persisted hidden bootstrap pair and the startup-stream adapter path that consumes it.
- The CLI/TUI path now also persists prompt history under the Python-style home cache history file, restores that history on startup for `Up`/`Down` recall, and excludes slash commands from recalled history instead of treating them as ordinary prompts.
- Local command result presentation now matches Python more closely: toast-target tool results only stay transient for palette-launched commands, while slash-launched commands and slash-opened command-form submissions now write their tool result into conversation history instead of incorrectly using the toast path.
- Background local-command status is now also source-neutral instead of slash-branded: the TUI enters a generic `running command...` state while the worker is active, ordinary non-toast command completions no longer synthesize a `slash command executed: /...` status, and tool-layer failures now surface as command failures without pretending every local command came from slash input.
- Persisted transcript rendering is now a little closer to Python too: system messages stay hidden in the TUI snapshot path, and persisted tool-role messages render as `tool result: ...` in both initial snapshot loads and background context-poll appends instead of showing a raw `tool: ...` prefix.
- Persisted assistant transcript rendering is now a little closer to Python too: hidden `<internal_thought>...</internal_thought>` segments are stripped from both snapshot loads and background context-poll appends instead of leaking those raw tags into the visible conversation pane.
- The Python-style `show_internal_thought` toggle now also exists in the Rust config/runtime path: it loads from file/env config, stays hidden by default, and when enabled it renders live and persisted assistant thought segments into the TUI conversation pane as plain-text `thinking: ...` lines instead of dropping them entirely.
- The Rust TUI chat composer now supports basic in-line editing instead of being append-only: characters insert at the cursor, `Backspace`/`Delete` edit around the cursor, the terminal cursor is positioned inside the input box, paste inserts at the cursor, and real `Left`/`Right` key events now work through the live event path again, including sidebar section switching while command-mode sidebar focus is active.
- Prompt-active footer rendering is now a little closer to Python’s Textual worker status too: active chat streams use the Python braille spinner sequence instead of a static status line, while background and command-action footer behavior stays unchanged.
- Manual conversation scrolling now stays respected across new streamed prompt output and background context-poll appends even while input focus is retained, instead of snapping the view back to the latest line just because the user was not in explicit conversation-browse focus.
- The first `Escape` out of chat mode now mirrors Python’s `toggle_browse` default too: browse mode enters the sidebar first, not the conversation pane, while repeated `Tab` / `Shift+Tab` still cycle between sidebar and history and the last non-chat target is remembered when returning from chat mode.
- The wrapped chat composer now also matches Python’s height cap more closely: it still grows for multi-line drafts, but it stops at the Python-style 8-row maximum instead of continuing to expand until it consumes most of the terminal body.
- Command-palette filtering is now less literal and a little closer to Python’s matcher-driven behavior: title-prefix and title-substring matches are preferred, but non-contiguous fuzzy matches still remain selectable instead of requiring exact contiguous substrings everywhere.
- `Ctrl+C` handling now matches Python’s key priority more closely: when the chat input is focused and contains text, it clears that draft first, even if a prompt stream is active; pressing `Ctrl+C` again with an empty draft still cancels the active stream.
- Sidebar section switching now preserves Python-style per-section selection state: each sidebar list remembers its selected row when switching away and restores it when switching back, while refreshed sidebar snapshots clamp saved selections if a list shrinks.
- Validation for this slice: `cargo test -p elroy-config loads_yaml_config_and_ignores_unknown_keys -- --nocapture`, `cargo test -p elroy-config environment_overrides_file_values -- --nocapture`, `cargo test -p elroy-tui prompt_spinner_advances_only_while_prompt_is_active -- --nocapture`, `cargo test -p elroy-tui footer_status_text_prefers_active_status_during_prompt -- --nocapture`, `cargo test -p elroy-tui streamed_output_does_not_resume_following_after_manual_input_scroll -- --nocapture`, `cargo test -p elroy-tui background_context_updates_do_not_resume_following_after_manual_input_scroll -- --nocapture`, `cargo test -p elroy-tui escape_toggles_between_chat_and_last_command_pane -- --nocapture`, `cargo test -p elroy-tui command_mode_tab_toggles_between_conversation_and_sidebar -- --nocapture`, `cargo test -p elroy-tui command_mode_conversation_keys_scroll_history_instead_of_sidebar -- --nocapture`, `cargo test -p elroy-tui escaping_from_conversation_browse_reenables_following_latest_output -- --nocapture`, `cargo test -p elroy-tui input_box_height_grows_for_wrapped_text -- --nocapture`, `cargo test -p elroy-tui input_box_height_caps_at_python_max_height -- --nocapture`, `cargo test -p elroy-tui input_box_height_keeps_body_visible_on_short_terminal -- --nocapture`, `cargo test -p elroy-tui command_palette_filters_entries_from_typed_query -- --nocapture`, `cargo test -p elroy-tui command_palette_fuzzy_matches_non_contiguous_query -- --nocapture`, `cargo test -p elroy-tui command_palette_prefers_title_prefix_match_over_description_match -- --nocapture`, `cargo test -p elroy-tui left_right_key_events_switch_sidebar_sections_when_sidebar_is_focused -- --nocapture`, `cargo test -p elroy-tui chat_input_supports_cursor_insertion_and_deletion -- --nocapture`, `cargo test -p elroy-tui multiline_paste_is_flattened_in_chat_input -- --nocapture`, `cargo test -p elroy-tui chat_input_up_down_cycles_prompt_history -- --nocapture`, `cargo test -p elroy-tui apply_key_event_appends_input_and_submits_prompt -- --nocapture`, `cargo test -p elroy-tui internal_thought_prompt_updates_append_to_conversation_when_enabled -- --nocapture`, `cargo test -p elroy-tui poll_context_updates_hides_internal_thought_segments_in_assistant_messages -- --nocapture`, `cargo test -p elroy-tui poll_context_updates_can_render_internal_thought_segments_when_enabled -- --nocapture`, `cargo test -p elroy-app load_snapshot_formats_persisted_tool_messages_and_skips_system_lines -- --nocapture`, `cargo test -p elroy-app load_snapshot_can_render_internal_thought_segments_when_enabled -- --nocapture`, `cargo test -p elroy-tui -- --nocapture`, `cargo fmt --all`, and `cargo clippy -p elroy-config -p elroy-app -p elroy-tui --all-targets --all-features`.

## Phase 3: Memory And Reminder Quality

Goal:

- improve the product differentiators that matter most to actual Elroy usefulness: recall, reminders, and context maintenance

Why this comes second:

- the base mechanics already exist
- current behavior is functional but still heuristic and visibly lower quality than Python in the most important smart features
- this work increases product usefulness more than another round of tool-surface cleanup

Primary parity rows to advance:

- `elroy/repository/memories/`
- `elroy/repository/reminders/`
- `elroy/repository/tasks/`
- `elroy/repository/context_messages/`
- `Memory recall quality`
- `Reminder and due-item surfacing`

Focus areas:

1. Improve recall selection quality.
   - move beyond simple token-overlap heuristics
   - port more of the Python classifier/selection behavior
   - add scenario-driven parity tests instead of only narrow helper coverage
2. Improve context refresh quality.
   - replace deterministic synthetic summary quality with Python-like LLM-generated summary behavior
   - tighten scheduling/orchestration around refresh
3. Improve reminder surfacing quality.
   - richer contextual selection behavior
   - better interplay between due items, tasks, and current context
4. Improve memory lifecycle quality.
   - richer consolidation behavior
   - better source metadata and reflective recall behavior where Python has it

Exit criteria:

- memory recall feels predictably useful in realistic conversations
- reminders surface at the right times with fewer false negatives and fewer brittle heuristics
- long conversations degrade gracefully because context refresh quality is acceptable, not merely structurally present

Usable checkpoint:

- "Rust preserves the core Elroy differentiators" instead of merely reproducing CRUD surfaces

### Phase 3 Progress (2026-05-25)

- The repository-side Python `augment_text` helper is now present in Rust as `elroy_recall::augment_text_from_config(...)` instead of being an unported memory-quality gap.
- That path reuses the existing recall selectors over active memories and due items, asks the configured model to enrich the note text only when relevant context exists, and otherwise returns the original text unchanged.
- Direct scenario coverage now mirrors the Python memory tests for both the relevant-memory augmentation case and the no-relevant-memory passthrough case.
- The repository-side Python `ingest_memo` helper is now also present in Rust as `elroy_memory::ingest_memo_from_config(...)`, converting freeform note text into either a pinned memory or a due item through the normal file-backed stores.
- The Rust helper now also matches Python’s retry behavior for invalid due-item proposals: if the model first returns a past-due reminder, the retry prompt includes that failure and can succeed on a follow-up attempt instead of failing immediately.
- Direct scenario coverage now exists for both the memory-creation path and the invalid-reminder-then-contextual-reminder retry path.
- Newly created due items now also persist an embedding record immediately when embedding config is available, instead of waiting for a later semantic-recall path to lazily backfill that cache.
- Direct crate-local coverage now proves `create_due_item` persists that embedding record on creation.
- Validation for this slice: `cargo fmt --all`, `cargo test -p elroy-recall augment_text_from_config -- --nocapture`, `cargo clippy -p elroy-recall --all-targets --all-features`, `cargo test -p elroy-memory ingest_memo_from_config -- --nocapture`, `cargo clippy -p elroy-memory --all-targets --all-features`, `cargo test -p elroy-agenda create_due_item_persists_embedding_when_embedding_config_is_available -- --nocapture`, and `cargo clippy -p elroy-agenda --all-targets --all-features`.

## Phase 4: Repository And Persistence Completion

Goal:

- close remaining persistence and repository-behavior gaps that affect data fidelity, rebuild behavior, and compatibility with Python-era data

Why this comes third:

- the app should already be usable before deeper storage completion
- these gaps matter for trust and long-term correctness, but are less visible than Phases 2 and 3 in first-use flows

Primary parity rows to advance:

- `elroy/db/`
- `Persistence compatibility with existing user data`
- `File-backed inspectable recallable content`
- repository rows still marked partial for memory/reminders/tasks/context

Focus areas:

1. Broader repository-level sync/rebuild behavior.
2. Remaining data-compatibility edge cases for Python-era files and derived state.
3. Narrower domain-schema and higher-level repository workflow completion.
4. Better invariants around rebuilds, tombstones, inactive history, and backfill.

Exit criteria:

- rebuilding derived state from an existing Python data set is trustworthy
- repository behaviors are no longer the main source of parity caveats in the matrix

Usable checkpoint:

- "Rust is safe to adopt on existing user data without hidden repository caveats"

### Phase 4 Progress (2026-05-25)

- The Python repository-side `get_memories(ctx, [ids])` helper is now present in Rust as `elroy_memory::get_memories_from_config(...)`, backed by a lower-level `elroy_db::load_memories_by_ids(...)` query helper.
- That Rust path now matches the Python test shape for selective memory-ID lookup, including the empty-list and missing-ID cases instead of leaving that repository read helper unported.
- Direct crate-local coverage now exists in both `elroy-memory` and `elroy-db` for the requested-ID lookup behavior and stable requested-order return shape.
- The Python repository-side context-message add/remove operations now also have Rust equivalents through `elroy_context::add_persisted_context_messages(...)` and `elroy_context::remove_persisted_context_messages(...)`, backed by transactional `elroy_db::append_context_messages(...)` / `remove_context_messages_by_ids(...)` helpers rather than whole-transcript replacement.
- Those DB helpers now also take an immediate SQLite transaction so concurrent append writers serialize cleanly around the `position` index instead of racing on load-modify-replace behavior.
- Direct crate-local coverage now exists for append/remove ordering, wrapper-level add/remove behavior, and concurrent append writers against a shared SQLite file.
- The Python memory read-store query helpers now also have Rust equivalents through `elroy_memory::get_memory_by_name_from_config(...)` and `elroy_memory::get_active_memories_from_config(...)`, instead of leaving those repository-level reads spread across lower-level DB and recall helpers only.
- Direct crate-local coverage now exists that those helpers return the active in-scope memories while excluding archived ones.
- The Python memory source-read surface now also has a first structured Rust repository helper via `elroy_memory::get_source_list_for_memory_structured_from_config(...)`, plus a thin `get_source_content_for_memory_text_from_config(...)` wrapper for the corresponding source-content retrieval path.
- Direct crate-local coverage now exists that those helpers return structured `("Memory", name)` lineage for consolidated memories and `("ContextMessageSet", id)` lineage for transcript-backed memories while preserving the expected source content lookup behavior.
- The Python reminder read/query helpers now also have Rust equivalents through `elroy_reminders::get_db_due_item_by_name_from_config(...)`, `get_active_due_items_from_config(...)`, `get_due_timed_items_from_config(...)`, `get_due_item_by_name_from_config(...)`, and `get_due_item_context_messages_from_config(...)`, instead of leaving that repository-level reminder read surface implicit in app/runtime assembly only.
- Direct crate-local coverage now exists for the key Python repository cases: active due-item listing, exact-name lookup, timed-due detection, future timed-item omission, contextual-only omission from timed-due results, and timed due-item synthetic context generation.
- The Python combined memory-query helper `get_relevant_memories_and_due_items(...)` now also has a structured Rust counterpart through `elroy_memory::get_relevant_memories_and_due_items_from_config(...)`, returning typed memory/due-item/agenda-item matches instead of only the formatted tool-report surface.
- Direct crate-local coverage now exists that the helper can return all three categories from one overlap-based repository query.
- The Python recall-query metadata helpers now also have Rust counterparts through `elroy_recall::get_recall_metadata(...)`, `is_item_in_context_message(...)`, `is_item_in_context(...)`, `is_memory_in_context_message(...)`, `is_memory_in_context(...)`, `is_agenda_item_in_context_message(...)`, and `is_agenda_item_in_context(...)`.
- Direct crate-local coverage now exists that those helpers recognize the real synthetic current-context payloads emitted by `context_memory_tool_messages(...)` and `context_due_item_tool_messages(...)`.
- The Python recall read-store `query_vector(...)` shape now also has Rust repository equivalents through `elroy_recall::query_memories_by_embedding(...)` and `elroy_recall::query_agenda_items_by_embedding(...)`, while the existing top-2 helpers for memories, due items, and plain agenda items now build on that shared ranked-query surface instead of only existing as standalone convenience wrappers.
- Direct crate-local coverage now exists that those ranked-query helpers preserve Python-style ordering across active rows and that the top-2 memory/due/agendum wrappers filter out the correct categories on top of the shared agenda ranking.
- The Python standalone recall-classifier surface now also has Rust equivalents through `elroy_recall::apply_memory_recall_heuristics(...)` and `should_recall_memory_from_config(...)`, with more specific heuristic reasoning for acknowledgments, greetings, and clarification-only prompts instead of only the earlier embedded boolean skip check.
- Direct crate-local coverage now exists for the Python-style short-message heuristic cases, the config-disabled path, and the model-backed classifier path over recent conversation context.
- The Python standalone memory-cluster consolidation surface now also has a first Rust repository entry point through `elroy_recall::MemoryCluster` plus `consolidate_memory_cluster_from_config(...)`, reusing the existing consolidation outputs/archive path instead of leaving consolidation only reachable through threshold-driven auto-memory orchestration.
- Direct crate-local coverage now exists that consolidating a duplicate cluster archives the source memories and leaves only the consolidated active memory, matching the key Python repository expectation for `consolidate_memory_cluster(...)`.
- Validation for this slice: `cargo fmt --all`, `cargo test -p elroy-memory get_memories_from_config_returns_requested_memory_ids -- --nocapture`, `cargo test -p elroy-memory memory_query_helpers_return_active_memories_in_scope -- --nocapture`, `cargo test -p elroy-memory source_helpers_return_structured_memory_and_context_sources -- --nocapture`, `cargo test -p elroy-memory relevant_recall_helper_returns_memory_due_item_and_agenda_item_matches -- --nocapture`, `cargo test -p elroy-reminders reminder_query_helpers_match_due_item_repository_cases -- --nocapture`, `cargo test -p elroy-recall -- --nocapture`, `cargo test -p elroy-db load_memories_by_ids_preserves_requested_order -- --nocapture`, `cargo test -p elroy-db append_and_remove_context_messages_preserve_order -- --nocapture`, `cargo test -p elroy-db append_context_messages_supports_concurrent_writers -- --nocapture`, `cargo test -p elroy-context persisted_context_messages_can_be_appended -- --nocapture`, `cargo test -p elroy-context persisted_context_messages_can_be_removed_by_message_identity -- --nocapture`, and `cargo clippy -p elroy-memory -p elroy-db -p elroy-context -p elroy-reminders -p elroy-recall --all-targets --all-features`.

## Phase 5: Codex And Operational Completion

Goal:

- finish the remaining agent workflow and operational product gaps so the Rust app covers the active Python operational surface

Why this comes fourth:

- core assistant usefulness matters before specialized operational workflows
- Codex already exists structurally, so the remaining work is interactive completion rather than initial bring-up

Primary parity rows to advance:

- `elroy/repository/codex_sessions/`
- `Codex workflow support in v1`
- `elroy/__main__.py`
- `elroy/config/`
- `elroy/tools`

Focus areas:

1. Broader interactive Codex UI workflows.
   - beyond read-only inspection
   - better session lifecycle control and visibility
2. Remaining operational command and config surface parity.
3. Final packaging and entrypoint expectations for a release-candidate workflow.

Exit criteria:

- Codex workflows are first-class enough that they are not treated as "exists, but use Python if you really need it"
- config and operational surface support the intended first release

Usable checkpoint:

- "Rust covers the operational workflows the active project actually relies on"

### Phase 5 Progress (2026-06-03)

- `list_codex_sessions` now accepts the Python-style `scope` argument, filtering to the home contrib repo for `scope="contrib"` and the current running Elroy repo for `scope="elroy"` while preserving the existing Rust `repo_path` filter when no scope is supplied.
- The same tool now matches Python's error contract for unknown scopes and non-positive limits instead of silently defaulting or accepting arbitrary scope strings.
- Direct Codex tool coverage now proves scoped listing, the existing explicit repo-path filter, and the invalid-scope/invalid-limit cases.
- Python-named `inspect_elroy_with_codex` and `edit_contrib_with_codex` tool entries now exist in Rust, including Python-style inspection/contrib prompt construction, optional log inclusion for inspection, minimal home-contrib repo bootstrap for contrib edits, async Codex persistence, and shared running/completion status handling.
- Direct Codex tool coverage now proves the inspection prompt/log contract and the named contrib launch path, including repo bootstrap and persisted prompt content.
- The larger Phase 5 Codex launch gap is now narrower: the named Rust tools still reuse the isolated worktree dispatch path, while Python inspection runs directly against the source tree and Python contrib edits run directly in the home contrib repo.

## Phase 6: Test And Parity Closure

Goal:

- convert the remaining broad `partial` rows into either `parity` or explicit intentional deltas with strong verification

Why this is last:

- broad end-to-end closure is more efficient after the main behavior gaps are actually closed
- otherwise the team risks writing large amounts of test scaffolding around still-moving behavior

Primary parity rows to advance:

- `tests/`
- all remaining `partial` system rows
- all remaining `partial` cross-cutting rows

Focus areas:

1. Scenario-driven parity tests for high-value workflows.
2. End-to-end validation across runtime, persistence, and TUI seams.
3. Matrix cleanup:
   - remove stale `partial` wording
   - record intentional deltas explicitly
   - mark truly-finished rows as `parity`

Exit criteria:

- every in-scope row in `PARITY_MATRIX.md` is either `parity` or `intentional delta`
- remaining intentional deltas are explicit, justified, and accepted

Usable checkpoint:

- release candidate for replacing the Python app by default

## Recommended Slice Order Inside The Next Phase

The next sequence should be:

1. structural refactoring — extract elroy-recall, elroy-context, elroy-reminders; expand domain crate tool modules
2. finish the remaining high-friction TUI/session/background workflow gaps
3. improve memory recall and context-refresh quality
4. improve reminder selection quality
5. close repository/persistence completion gaps
6. finish Codex interactive workflow parity
7. do broad end-to-end test and matrix closure

This is intentionally different from the recent pattern of spending many consecutive commits on narrow contract cleanup.

## Pull Request Heuristics

For the next stretch of work, a good PR or commit series should usually satisfy one of these:

- closes a broad `partial` note in a major parity row
- turns a user-visible workflow from "exists" into "daily usable"
- replaces heuristic behavior with higher-fidelity Python behavior
- adds scenario-driven verification for a differentiating workflow

A weak PR pattern for now is:

- adjusts one tool string or schema detail without moving a phase exit criterion
- adds isolated helper coverage without advancing a broader workflow
- adds coverage-only tests for already-working behavior outside of Phase 5

**Phase self-check:** Before starting a commit, ask "does this directly close a gap on the current phase checklist or advance a phase exit criterion?" If the honest answer is no — and the work is coverage-only for already-working behavior — stop and pick a different task.

## Definition Of Roadmap Success

This roadmap is successful if it changes team behavior from:

- "pick the next tiny parity mismatch"

to:

- "pick the next broad workflow gap that improves the product and shrinks the matrix in a meaningful way"

The rewrite is complete only when:

- every in-scope subsystem in [PARITY_MATRIX.md](/Users/tombedor/development/elroy-rs/PARITY_MATRIX.md) is `parity` or `intentional delta`
- the Rust app is usable as the default local Elroy client throughout the main workflows
- end-to-end validation supports that claim rather than only many isolated unit tests
