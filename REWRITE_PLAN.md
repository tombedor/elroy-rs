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
6. Tests accompany new behavior; coverage-only commits are Phase 5 work. Writing a test for already-implemented behavior does not advance the current phase. A commit that only adds tests — with no behavioral change — belongs in Phase 5 regardless of which phase you are in. During Phases 1–4, write tests when they verify a behavior change in the same commit. If several consecutive commits are coverage-only, treat that as a signal to stop and redirect toward the phase exit criteria.

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
- coverage-only commits for already-working behavior while Phase 1 exit criteria remain unmet — this is the single most common way to consume velocity without advancing the product

Those are acceptable as opportunistic cleanup inside a broader slice, but they should not drive roadmap order.

## Roadmap Phases

## Phase 1: Usable Core Product

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

## Phase 2: Memory And Reminder Quality

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

## Phase 3: Repository And Persistence Completion

Goal:

- close remaining persistence and repository-behavior gaps that affect data fidelity, rebuild behavior, and compatibility with Python-era data

Why this comes third:

- the app should already be usable before deeper storage completion
- these gaps matter for trust and long-term correctness, but are less visible than Phases 1 and 2 in first-use flows

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

## Phase 4: Codex And Operational Completion

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

## Phase 5: Test And Parity Closure

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

1. finish the remaining high-friction TUI/session/background workflow gaps
2. improve memory recall and context-refresh quality
3. improve reminder selection quality
4. close repository/persistence completion gaps
5. finish Codex interactive workflow parity
6. do broad end-to-end test and matrix closure

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
