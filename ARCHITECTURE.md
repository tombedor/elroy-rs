# Architecture

This document defines the target architectural shape for the Rust rewrite of Elroy.

It is intentionally stricter than a casual design note. Its job is to stop the rewrite from drifting into an unstructured port.

## Design Goals

- preserve Elroy's user-visible behavior
- keep dependency boundaries explicit
- make runtime lifetimes visible in the type system
- keep workflow logic separate from storage and presentation
- support incremental subsystem ports

## Proposed Workspace Shape

Exact crate names may change, but the boundaries should remain close to this shape:

- `elroy-cli`: CLI entrypoints and process startup
- `elroy-app`: shared app-service runtime wiring for prompt execution, snapshot loading, and UI-facing workflows
- `elroy-config`: config loading, defaults, env/file precedence
- `elroy-core`: session, turn/request context, runtime composition, orchestration primitives
- `elroy-llm`: model client abstraction, stream parsing, tool-call extraction
- `elroy-tools`: tool registry, schemas, tool execution interfaces
- `elroy-db`: DB connections, migrations, persistence infrastructure
- `elroy-memory`: memory lifecycle, recall, consolidation
- `elroy-reminders`: reminders and due-item workflows
- `elroy-agenda`: agenda item workflows
- `elroy-context`: context message read/refresh logic
- `elroy-user`: user preferences and user session data
- `elroy-codex`: Codex session support
- `elroy-tui`: terminal UI

It is acceptable to collapse some crates early if doing so reduces bootstrap cost, but responsibility boundaries must still be respected at the module level.

Current reality is now partially aligned with that target:
- `elroy-memory` owns the first file-backed memory operations
- `elroy-agenda` owns the first file-backed agenda operations
- `elroy-app` composes those domain crates into the live runtime and TUI/CLI workflows

## Runtime Lifetimes

The Python project distinguishes long-lived config/session state from per-turn state. The Rust port should preserve that separation more explicitly.

### AppConfig

Responsible for:

- long-lived application configuration
- model selection and tuning parameters
- UI and runtime configuration defaults
- paths and feature flags

Must not:

- own live DB transactions
- own request-local metadata
- become a service locator for every subsystem

### AppSession

Responsible for:

- stable user/session identity
- long-lived handles needed across many turns
- process-level state that survives individual requests

Must not:

- own request-local state
- absorb UI state and domain workflows indiscriminately

### TurnContext

Responsible for:

- one user turn or one tool-execution scope
- live DB transaction/session
- request ids, timing, cancellation, and per-turn tracing
- references to the active session and the config/runtime inputs needed for the turn

Must not:

- become a broad dump for unrelated state
- outlive the request boundary it represents

## Role Vocabulary

Adopt a small role vocabulary similar to the useful discipline in the Python repo.

### Store

Responsible for:

- persistence and retrieval for one domain entity or one tightly related entity set
- narrow invariants close to stored data
- translating between stored and domain representations

Must not:

- own cross-domain workflows
- coordinate multiple sibling components for a user action

### Orchestrator

Responsible for:

- end-to-end workflows across stores, builders, indexers, and runtime components
- transaction and side-effect ordering
- turning a user or system action into a multi-step workflow

Must not:

- own low-level persistence details
- become a generic bag of unrelated logic

### Builder

Responsible for:

- constructing derived artifacts from existing data
- prompt assembly
- summaries, transformed payloads, and read-oriented derivations

Must not:

- own persistence
- coordinate large workflows

### Indexer

Responsible for:

- maintaining derived search or retrieval structures
- embeddings and recall index maintenance
- keeping indexed artifacts aligned with source-of-truth state

Must not:

- become the primary owner of source entities
- own general workflow coordination

## Dependency Rules

- orchestrators may depend on stores, builders, indexers, and narrow runtime interfaces
- stores must not depend on orchestrators
- builders should stay pure or mostly pure
- indexers must not depend on orchestrators
- UI code must not depend directly on persistence stores for workflow execution
- composition roots may wire everything together, but inner modules should receive narrow dependencies

Bidirectional dependencies are a design smell.

## Async Model

The Rust implementation should assume async workflows for:

- model calls
- tool execution that may block on IO
- background tasks
- UI-to-runtime coordination where the selected TUI stack supports it

Async does not justify hiding boundaries. Blocking adapters should remain explicit where they exist.

## Persistence Model

This remains an open implementation choice, but the architectural rule is clear:

- source-of-truth ownership for each entity must be explicit
- if data is file-backed in Python for user-facing and recallable reasons, preserve that behavior unless a deliberate migration says otherwise
- database indexes, caches, and derived search state should not silently become source-of-truth stores
- humans should be able to inspect the same recallable information the agent relies on
- `elroy-rs` must be able to read the same compatible file set as the Python app and backfill required derived database state from it

Before implementing persistence-heavy slices, document:

- ownership of each entity
- migration path from Python data
- compatibility expectations
- what is canonical on disk versus what is rebuildable/derived

## UI Boundary

The TUI should preserve behavior described in `../elroy/UI.md` and related Python UI code.

UI components should:

- render state
- emit typed intents/actions
- delegate workflows to controllers or orchestrators

UI components should not:

- directly implement domain persistence workflows
- become the place where business logic accumulates

## Conversation Loop Boundary

The assistant loop is a core parity surface.

It should have clear seams for:

- loading context
- deciding whether recall should run
- appending memory and due-item context
- generating model output
- executing tool calls
- persisting resulting messages and side effects

This path should be testable without a full TUI boot.

## Testing Consequences

The architecture should support:

- pure unit tests for builders and decision logic
- store tests against realistic persistence backends
- orchestrator tests for workflow behavior
- snapshot or golden tests for stream/event ordering
- UI tests for key interaction flows where practical

The rewrite should favor tests as the primary proof of parity. Test documentation should make clear which Python behaviors are being validated and which compatibility guarantees each suite covers.

## Decision Logging

When a major architectural choice is made, record it in a dedicated decision doc or ADR set.

At minimum, decisions should be recorded for:

- async runtime
- TUI framework
- persistence stack
- migration tooling
- vector search / embedding strategy

## Current Working Decisions

These decisions are settled enough to guide implementation now:

- Async-first is the default runtime model
- `ratatui` is the selected TUI framework
- SQLite-first persistence is the target, with `rusqlite` plus `refinery` as the working direction and `tokio-rusqlite` available at async boundaries if needed
- Recallable and user-inspectable content should remain file-backed where the Python implementation uses files as the visible source of truth
- Derived database/index state must be rebuildable from a compatible Elroy file set
- Tool/model behavior should remain close to the current Python implementation and stay interoperable with Anthropic and OpenAI models via a canonical internal tool schema plus provider-specific adapters
- Codex workflow support is required for the first Rust version

These remain open:

- *(none — all major structural decisions are now settled)*

## Decision Log

### Crate boundary approach

**Decision:** Aggressive structural refactoring now (REWRITE_PLAN.md Phase 1) rather than waiting for fuller parity.

**Rationale:** `elroy-app` grew into a 28K-line monolith absorbing tool execution, recall orchestration, consolidation, context refresh, and reminder surfacing. Phase 2+ improvements to recall and context-refresh quality require editing in an isolated, testable crate — not a file that also contains tool routing, formatting, and prompt pipeline logic.

**Implementation:** Phase 1 extracts `elroy-recall`, `elroy-context`, and `elroy-reminders` from `elroy-app`, and expands domain crates with tool execution modules. `elroy-app` drops to under 8K LOC.

**Success criterion:** `cargo build && cargo test` clean; `elroy-app` under 8K LOC; no behavioral changes.

## Tool Execution Pattern

Each domain crate that implements tools exports two functions:

```rust
pub fn tool_specs() -> Vec<ToolSpec>;
pub async fn execute(name: &str, args: &JsonValue, ctx: &ToolContext) -> Result<ToolExecutionResult>;
```

`elroy-app` maintains a single `dispatch_tool` function that routes by tool name to the owning crate. This routing table is the only place in `elroy-app` that knows which crate owns which tool.

Tool specs at startup are composed from each domain crate's `tool_specs()`. Adding a tool to a domain crate should not require changes to `elroy-app` beyond one entry in the routing match.

## Config Injection Pattern

Orchestrators must not receive the full `AppConfig`. Each orchestrator takes only the config sub-struct it needs:

```rust
// Wrong:
fn new(config: &AppConfig) -> RecallOrchestrator

// Correct:
fn classify_recall_needed(msg: &str, ctx: &[ContextMessage], cfg: &RecallConfig) -> bool
```

Each orchestrator crate defines (or re-exports from `elroy-config`) its own narrow config type. `elroy-app` extracts the relevant sub-config and passes it down. This makes dependencies explicit and prevents `AppConfig` from becoming a service locator.
