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

### New Crates To Create

**`elroy-recall`**

Owns all memory recall and consolidation logic currently embedded in `elroy-app`:

- recall classification (heuristic and model-backed)
- candidate selection and expansion (lexical overlap, embedding ranking)
- relevance filtering (model-backed)
- reflective recall generation (deterministic and model-backed)
- exact-duplicate and semantic-cluster consolidation
- embedding cache management
- auto-memory creation from context

`elroy-memory` becomes a pure file I/O + frontmatter Store. All orchestration moves to `elroy-recall`.

**`elroy-context`**

Owns all context message loading and refresh logic currently embedded in `elroy-app`:

- transcript loading and validation (role alternation repair, orphaned tool call repair)
- system message building and repair
- context compression and summary generation (deterministic and model-backed)
- context refresh scheduling and orchestration
- due-item and reminder pinning into transcript context

This crate already exists in `ARCHITECTURE.md`'s target shape; it just hasn't been created yet.

**`elroy-reminders`**

Owns due-item surfacing workflows currently embedded in `elroy-app`:

- due-item context message generation
- synthetic tool message creation for surfaced reminders
- reminder selection heuristics
- interplay between due items, tasks, and current context

This crate already exists in `ARCHITECTURE.md`'s target shape; it just hasn't been created yet.

### Domain Crates That Expand

Each of these crates gains a `tools` module owning its own tool execution and tool specs. The corresponding match arms and execution functions move out of `elroy-app`.

| Crate | Tools to absorb from elroy-app |
|---|---|
| `elroy-memory` | `create_memory`, `show_memory`, `search_memories`, `print_memories`, `update_memory`, `archive_memory` |
| `elroy-agenda` | `create_agenda_item`, `update_agenda_item`, `complete_agenda_item`, checklist tools |
| `elroy-tasks` | `create_task`, `show_task`, `list_tasks`, `update_task`, `complete_task` |
| `elroy-user` | `update_user_preferred_name`, `update_assistant_name`, user preference tools |
| `elroy-feature-requests` | `create_feature_request`, `list_feature_requests`, `show_feature_request`, `update_feature_request` |
| `elroy-codex` | `dispatch_codex_session`, `resume_codex_session`, `list_codex_sessions` |

### `elroy-tools` Expands

`elroy-tools` gains base tool implementations currently embedded in `elroy-app`:

- filesystem tools: `ls`, `read_file`
- developer tools: `get_help`, `print_config`, `tail_elroy_logs`, `restart_session`
- context tools: `reset_messages`, `refresh_system_instructions`

### `elroy-app` After Refactoring

`elroy-app` becomes a thin wiring and routing layer only:

- `AppRuntime` construction and dependency composition
- tool execution routing (dispatching by tool name to domain crate executors)
- snapshot loading (composing from domain crates)
- command palette and command form prefilling
- `process_message` / `load_snapshot` / `load_context_messages` surface API for TUI/CLI

Target size: under 8K LOC (down from ~28K).

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

- [ ] `elroy-recall` created; all recall and consolidation logic moved from `elroy-app`
- [ ] `elroy-context` created; transcript loading, validation, and refresh orchestration moved from `elroy-app`
- [ ] `elroy-reminders` created; due-item surfacing moved from `elroy-app`
- [ ] Each domain crate owns its tool execution; match arms removed from `elroy-app`
- [ ] `elroy-tools` owns base/filesystem/developer tools
- [ ] `elroy-app` is under 8K LOC
- [ ] All existing tests pass with no behavioral changes
- [ ] `cargo build` and `cargo test` are clean

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

- [ ] Broader session workflows in the TUI — concretely: decide whether Phase 1 still needs anything beyond the already-ported greeting-on-fresh-start and restart/session transitions, or whether the remaining work is truly multi-session history/session switching and can wait for a later phase
- [x] Deeper command-form validation parity — required fields now validate before final submit at the TUI layer instead of only surfacing a missing-value failure on `Enter`
- [x] Fuller Textual-style command-palette system-command behavior — Python’s surfaced system-command set (`Focus Memories`, `Focus Agenda`, `Refresh System Instructions`, `Reset Messages`) is now present in the Rust palette, with additional Rust-only section focus entries documented as intentional extensions
- [x] Broader background-status producers — the shared Rust footer now covers the real long-running background paths in use (`context-refresh`, `self-reflection`, `auto-memory`, background command execution, Codex dispatch/resume plus completion follow-up); the remaining Python-only worker groups (`session-bootstrap`, `sidebar-refresh`) are foreground or synchronous flows in the current Rust architecture rather than missing shared background producers

Focus areas:

1. Finish the remaining session workflow gaps in the TUI.
   - full restart/session transitions
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
