# Stage 2 — Real-Lab Adapter (φ / ψ / η)

Inference-stage adapter that lets the **abstract-trained** Pentest-PRM (Stage 1) operate on
**real single-host targets**. It maps concrete tool output onto the same `Observation` / `Action`
schema the oracle + PRM were trained on. See `../STAGE2_PLAN.md` for the full plan.

> **Safety.** This package performs **no live execution** by default. `eta.LiveExecutor` refuses to
> run unless an explicit authorization env var is set *and* an isolated-lab confirmation flag is
> passed. Phase 1 (implemented here) is **offline replay only** — it parses recorded tool output.
> Live exploitation (Phase 2/3) stays gated behind an owned, isolated, authorized lab.

## The three maps

| map | direction | status | module |
|---|---|---|---|
| **φ (phi)** | real tool output → `Observation` | NEW | `phi.py` |
| **ψ (psi)** | LLM action text → `Action` | Stage-1 normalizer + Stage-2 coverage layer | `web_attack_sim/normalizer.py`, `psi.py` |
| **η (eta)** | abstract action → concrete command | NEW (templates + gated executors) | `eta.py` |

### ψ coverage layer (`psi.py`) — closes the Phase-1 bottleneck

Phase 1 found the abstraction covers ~92% of real steps but the Stage-1 keyword normalizer maps only
**49%** of natural operator phrasing (false-rejecting 32%). `psi.EnhancedNormalizer` wraps the **frozen**
Stage-1 ψ (Stage 1 untouched): on a Stage-1 `unsupported` result it runs an **out-guard** (keeps SSH /
offline-crack / `su` / kernel / binary-overflow OUT) then a general verb×object **recovery** for the
in-scope intents. Tuned only on the independent `psi_benchmark.jsonl`; measured on the walkthroughs as a
**held-out** test → ψ accuracy **49%→78.5%**, false-reject **32%→3%**, **false-accept stays 0**.

```powershell
python -m stage2.eval_psi --report-output outputs\stage2_psi_eval.json   # baseline vs enhanced, dev + held-out
python -m stage2.replay --enhanced-psi --report-output outputs\stage2_phase1_report_enhanced.json
```

The frozen strong PRM was trained on Stage-1-ψ features, so `--enhanced-psi` uses the enhanced layer only
for the coverage/execution mapping; PRM candidate features stay on the training-time ψ (rerank top-1
preserved). The `--prm-uses-enhanced-psi` diagnostic shows that swapping the PRM's normalizer too pushes
it out-of-distribution (rerank 0.5→0.1) — putting enhanced ψ in the PRM feature path needs a PRM retrain.

Closed loop: `propose candidates → ψ normalize → PRM rerank → η execute → φ observe → repeat`.

## Phase 1 — the decisive measurement (offline, safe)

Replays recorded box walkthroughs through φ + ψ + the strong PRM and measures, against per-step
**hand-labels**:

- **out-of-abstraction rate** — fraction of real steps with no abstract action (the make-or-break
  abstraction-gap number; ground truth, not ψ-inferred).
- **ψ accuracy / false-reject** — does the normalizer map the mappable steps correctly.
- **φ field recall** — does the parser reconstruct the abstract-state fields each step established.
- **PRM rerank top-1 / pairwise** — does the abstract-trained PRM prefer the good real action.

```powershell
# validate one fixture (schema + φ parse coverage)
python -m stage2.validate_fixture stage2\walkthroughs\dc-1.json

# run the full Phase-1 replay + decision gate over all fixtures
python -m stage2.replay --walkthroughs stage2\walkthroughs --report-output outputs\stage2_phase1_report.json

# offline closed-loop proxy (propose→ψ→PRM→η→ReplayExecutor→φ)
python -m stage2.closed_loop --walkthroughs stage2\walkthroughs --report-output outputs\stage2_closed_loop.json

# assemble the markdown report
python scripts\build_stage2_report.py
```

**Decision gate:** if the out-of-abstraction rate exceeds 60%, do a targeted schema extension (the
report prints a prioritized shortlist of missing capabilities) *before* building η / Phase 2.

## Walkthrough fixtures

`walkthroughs/*.json` — one recorded successful chain per box. Each step carries the operator
intent (ψ input), the real tool + raw output (φ input), a hand-labeled ground-truth abstract action
(one of 16 or `out_of_abstraction`), the abstract-state fields it establishes, and an optional
`candidate_pool` for the PRM rerank test. Fixtures are **author-constructed from public write-up
structure** (the `source` field says so) — they measure *structural* schema/normalizer/parser
coverage, not live value-uplift (that is Phase 2/3). See `fixtures.py` for the schema.

## Phase 2/3 readiness (built + gated — one authorization away from live)

The full live-capable stack is implemented and tested; it only refuses to *run* until authorized.

| piece | module | what it does |
|---|---|---|
| safety harness | `safety.py` | `AuthorizationGate` (env confirm + `confirmed_isolated` + kill-switch), lab-target scoping (private/loopback/.lab only — public refused), command allow-list + destructive denylist, JSONL `AuditLog` |
| η executor | `eta.py` | `LiveExecutor` now actually runs (subprocess, tokenised, timeout, audited) **only past the gate**; `DryRunExecutor` renders+logs nothing-executed; `ReplayExecutor` walks a fixture |
| target descriptors | `targets/*.json` | per-target η fills (DVWA recommended-first; VulnHub example) |
| engagement runner | `engagement.py` | the real loop `propose→ψ→PRM rerank→η→executor→φ`, A/B (`prm` vs `llm_only`), `StateProposer` (offline) / `LLMProposer` (DeepSeek, gated), budget + kill-switch + audit |
| preflight | `preflight.py` | one command → "are we ready?": verifies artifacts, η coverage, ψ held-out, gate deny-by-default, then lists the operator-only live prerequisites |

```powershell
python -m stage2.preflight                                           # OFFLINE READY: True (everything checkable)
python -m stage2.engagement --executor dryrun --mode ab              # exercise the loop, nothing runs
python -m stage2.engagement --executor replay --fixture stage2\dryrun\dvwa.json --mode ab
```

Going live is the **`STAGE2_PHASE2_RUNBOOK.md`** sequence (authorize → validate φ/ψ on real logs →
DVWA closed loop → A/B across a small box set). Nothing executes against a target without the env
confirmation string **and** `--confirmed-isolated` **and** a private/.lab target.
