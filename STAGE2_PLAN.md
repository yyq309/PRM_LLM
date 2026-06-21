# Stage 2 — Real-Lab Adapter Validation (Plan)

Stage 1 (training) is complete and frozen. Stage 2 answers **one question**: does the
abstract-trained Pentest-PRM give **real uplift** on **real targets**, and **how much does the
16-action abstraction miss**? It does NOT retrain the oracle/PRM — those are inputs.

> Scope/safety gate: real exploitation only against **owned, isolated, authorized** lab targets,
> with snapshot/restore and full logging. No internet targets.

> **Status (2026-06-17).** Phase 0 (safety harness) + Phase 1 (offline φ/ψ replay) are **done**, and
> the live-capable Phase-2 stack is **built and gated** — see `STAGE2_PHASE1_REPORT.md`,
> `STAGE2_PHASE2_RUNBOOK.md`, and `python -m stage2.preflight` (OFFLINE READY: True). Phase-1 verdict:
> abstraction gap only **8.5%** (no schema extension), ψ normalizer lifted **49%→78.5%** held-out via
> `stage2/psi.py`. What remains is operator-gated: authorization + an isolated lab + (recommended) a
> few real recorded logs, then the live DVWA→A/B runbook. Nothing here executes until then.

## 0. What Stage 1 hands over (ready)
- **ψ (normalizer)** — natural-language LLM action → one of the 16 abstract actions, with a measured
  real out-of-schema rate (~49% on DeepSeek). **Reused as-is.**
- **PRM (reranker)** — scores/ranks candidate abstract actions from observable state+action features.
- **oracle** — optional value prior (`Q/V/value_gap`).
- All with honest metrics + a one-command reproducer.

## 1. The adapter trio (the new build)
| map | direction | what it does | reuse? |
|---|---|---|---|
| **φ (phi)** | real tool output → `AbstractWebState` | parse nmap/gobuster/ffuf/sqlmap/whatweb/curl/burp output into discovered_paths, forms, parameters, credentials, auth/shell/privilege state, verified_vulns, read_files | **NEW** |
| **ψ (psi)** | LLM action text → `AbstractWebAction` | the Stage-1 normalizer | **DONE** |
| **η (eta)** | abstract action → concrete command | turn `web_path_enumeration /admin` into `gobuster dir -u T/admin`, `exploit_attempt` into the right sqlmap/curl call, etc. (sandboxed) | **NEW** |

Closed loop: LLM proposes candidates → **ψ** normalize → **PRM** rerank → **η** execute (sandboxed)
→ **φ** observe → repeat.

## 2. Phased milestones (de-risk cheap→expensive)
- **Phase 0 — harness & safety (infra).** Authorization, isolated Docker network, VM snapshot/restore,
  structured logging, target selection. No mapping yet.
- **Phase 1 — φ/ψ OFFLINE replay (decisive, no execution).** Take recorded walkthroughs / saved tool
  logs of the target boxes; run φ+ψ to map them to abstract state+action **without executing anything**.
  → measures **adapter mapping accuracy** and the real **out-of-abstraction rate**. Cheap, safe, and it
  answers the make-or-break abstraction-gap question before any execution investment.
- **Phase 2 — η executor + single-box closed loop.** Wire η (sandboxed tool commands) and run the full
  loop on ONE easy box (DVWA). Goal: the loop reaches a flag with PRM reranking.
- **Phase 3 — A/B uplift.** Small box set (DVWA, Juice Shop, 1–2 easy VulnHub). Compare
  **LLM + PRM-rerank vs LLM-only** on goal-reach rate / steps / wasted actions.
- **Phase 4 — report.** Real uplift, out-of-abstraction rate, failure taxonomy, schema-gap priorities.

## 3. The first measurable experiment (Phase 1)
- Input: N recorded successful walkthroughs (HTTP logs + tool output) for the chosen boxes.
- Output metrics: (a) **mapping accuracy** = fraction of real states φ reconstructs correctly vs a
  hand-labeled reference; (b) **out-of-abstraction rate** = fraction of real actions ψ cannot map;
  (c) per-tool φ coverage. No live exploitation — pure parsing.
- **Decision gate:** if out-of-abstraction rate is high (e.g. >60%), do a **targeted schema extension**
  (add the top missing action types — SSTI/SSRF/deserialization/robots/source-view) BEFORE Phase 2.
  (Cheaper to learn this offline than from live execution.)

## 4. What's needed ("需要什么")
**Authorization & safety**
- [ ] Written authorization + scope for the chosen lab targets (owned/CTF only).
- [ ] Isolated lab network; VM/container snapshot-restore; kill-switch; full audit logging.

**Infrastructure**
- [ ] Docker host + the target containers: DVWA, OWASP Juice Shop, WebGoat; 1–2 easy VulnHub boxes.
- [ ] A sandboxed execution wrapper for η (resource/网络-confined, reversible).

**Tooling for η / φ**
- [ ] Tool set: gobuster/ffuf, sqlmap, curl, whatweb/nikto, a controlled webshell handler.
- [ ] φ parsers (one per tool's output format) → AbstractWebState.
- [ ] η command templates (one per abstract action) → concrete tool invocation.

**Models / data**
- [ ] The agent LLM (DeepSeek V4; key only in `$DEEPSEEK_API_KEY`).
- [ ] Stage-1 artifacts: normalizer (ψ), `prm_strong.joblib` (reranker), canonical oracle (prior).
- [ ] Recorded ground-truth walkthroughs for Phase-1 mapping accuracy.

**Engineering**
- [ ] Closed-loop runner (propose → ψ → PRM rerank → η → φ → repeat) with budget + safety stops.
- [ ] Telemetry: out-of-abstraction events, mapping confidence, PRM ranks, outcomes.

## 5. Risks (grounded in Stage-1 evidence)
1. **Abstraction wall**: ~49% of real LLM actions are out-of-schema → PRM only helps the in-schema half.
   Mitigation: Phase-1 measures it FIRST; schema-extend only if needed.
2. **PRM is a ranker, not a policy** (closed-loop autonomous = no lift): use it to **rerank** the LLM's
   candidate next-steps, not to drive the episode alone.
3. **φ is lossy / real tool output is messy** → mapping accuracy < 1; report it honestly.
4. **sim-to-real value gap is unmeasured** — Stage 2 IS the measurement; expect modest, not dramatic, uplift.
5. **Safety** — real exploitation must stay sandboxed, authorized, reversible.

## 6. Success criteria
- Phase 1: out-of-abstraction rate + mapping accuracy quantified; schema-extension decision made.
- Phase 3: a statistically honest **A/B** number (with CIs) for PRM-rerank uplift on real boxes,
  reported alongside the out-of-abstraction rate and a failure taxonomy — positive OR negative, honestly.

## 7. Recommended first step
**Phase 0 + Phase 1 only.** They need no live exploitation (just infra + offline log parsing + the
existing ψ), they are cheap and safe, and they decisively answer the abstraction-gap question that
gates everything downstream. Decide on schema extension from Phase-1 numbers before building η.
