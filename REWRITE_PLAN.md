# Rewrite Plan

This document defines the migration strategy for porting `../elroy` to Rust.

## Objective

Build a Rust implementation of Elroy that achieves functional parity with the current Python application in controlled, testable increments.

Parity default:

- strict parity for core workflows unless a delta is explicitly documented and accepted

## Scope

In scope:

- CLI and TUI behavior
- conversation orchestration
- model integration surface
- tool registration and execution
- memory, reminders, agenda, and related repositories
- persistence and migration strategy
- Codex session and agent workflow support present in the Python project
- bootstrap/reconciliation from an existing Python-era file set

Out of scope during parity phase:

- major product redesign
- new feature families not already present in Python
- UI rethinks that change established interaction behavior

## Migration Strategy

Port in dependency order, not by file count.

The preferred approach is:

1. establish workspace and core boundaries
2. port low-level runtime and config primitives
3. port domain slices behind tests
4. port UI and end-to-end interaction once lower layers are stable

## Milestones

## Current Progress

- `M0` workspace bootstrap: completed
- `M1` core runtime and session model: started
- `M2` persistence and domain data model: started at the bootstrap-planning level
- `M5` TUI and UX parity: started with a `ratatui` shell

### M0: Workspace Bootstrap

Goals:

- create Cargo workspace structure
- define standard commands for format, lint, test, and run
- choose baseline libraries through recorded decisions
- establish fixture and test harness layout

Done when:

- the repo builds
- the repo tests run
- agent workflow docs match the actual command surface

Status:

- completed on 2026-05-13
- workspace commands validated with `just fmt`, `just lint`, `just test`, and `just run`

### M1: Core Runtime And Session Model

Goals:

- define config, session, and turn/request boundaries
- define runtime composition objects
- define error types, logging, and cancellation model
- define model client abstraction and tool execution interfaces

Source areas:

- `../elroy/elroy/core/`
- `../elroy/elroy/config/`

Done when:

- core boundaries are documented and implemented
- representative unit tests exist for config/session/runtime behavior

### M2: Persistence And Domain Data Model

Goals:

- define Rust data model for persisted entities
- choose migration strategy and schema tooling
- implement stores/repositories for core entities
- define compatibility expectations with existing Elroy data
- preserve file-backed source-of-truth behavior for user-inspectable recallable content where the Python app does so
- support database/index backfill from an existing Elroy file set

Source areas:

- `../elroy/elroy/db/`
- `../elroy/elroy/repository/data_models.py`
- repository store/query modules

Done when:

- the primary entities can be persisted and loaded in Rust
- migration/document compatibility rules are documented
- a bootstrap path exists to derive required DB state from compatible files

### M3: Tool System And Conversation Loop

Goals:

- implement tool registry and schema generation
- implement assistant loop with tool calls
- port status and streaming event model
- preserve execution ordering and persistence boundaries

Source areas:

- `../elroy/elroy/tools/`
- `../elroy/elroy/core/conversation_orchestrator.py`
- `../elroy/elroy/llm/`

Done when:

- conversation turns can execute with tools and persist resulting context
- parity tests cover the main loop behavior

### M4: Memory, Recall, Reminders, Agenda

Goals:

- port memory storage and retrieval behavior
- port recall classification and context injection
- port reminders, due items, agenda, and task workflows
- preserve file-backed and database-backed responsibilities where intended

Source areas:

- `../elroy/elroy/repository/memories/`
- `../elroy/elroy/repository/reminders/`
- `../elroy/elroy/repository/agenda/`
- `../elroy/elroy/repository/tasks/`
- `../elroy/elroy/repository/context_messages/`

Done when:

- recall and due-item behaviors are tested against Python expectations
- the main product differentiators work end-to-end

### M5: TUI And UX Parity

Goals:

- implement terminal UI layout and command flow
- preserve keybindings, focus behavior, and streaming output semantics
- preserve sidebar and modal behaviors

Source areas:

- `../elroy/elroy/ui/`
- `../elroy/UI.md`
- `../elroy/ROADMAP.md` UI section

Done when:

- core keyboard workflows match the current app
- user-visible behavior is covered by UI or end-to-end tests where feasible

### M6: Agent Workflow And Operational Polish

Goals:

- port Codex session support and related workflows
- complete operational commands and packaging
- document intentional post-parity improvements

Source areas:

- `../elroy/elroy/repository/codex_sessions/`
- roadmap and operational docs

Done when:

- the Rust app covers the current operational feature set required by active Elroy workflows
- Codex workflow support is present in the first release candidate, not deferred to a later version

## Sequencing Rules

- Do not start broad UI work before core runtime and domain behavior are stable enough to drive it
- Do not mark a subsystem complete without updating `PARITY_MATRIX.md`
- Prefer one partially vertical subsystem over many disconnected stubs
- If a major dependency choice affects several milestones, record it in architecture docs immediately

## Completion Criteria

The rewrite is ready to replace the Python app only when:

- every in-scope subsystem is at `parity` or `intentional delta`
- intentional deltas are documented and accepted
- the default workflows are validated end-to-end
- build, lint, and test workflows are stable for contributors and agents

## Open Questions

- What persistence stack best balances SQLite compatibility, migrations, and async ergonomics?
- Which parts of the Python file-backed model should remain file-backed versus become database-owned in Rust?
