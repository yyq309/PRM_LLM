# Contributions & Paper Main-Line (converged 2026-06-22)

This is the **canonical framing** for the paper. It supersedes both (i) the earlier "proposer-conditional
value / proposer-competence crossover" framing and (ii) the "abstract→real value transfer" spine. The
current spine names the **object** (a process-reward evaluator for penetration testing) and places the
novelty in **how we obtain it label-free and prove it works end-to-end on real machines.**

Two framing decisions, both honesty-driven:
- **Proposer/prompt quality is NOT a contribution.** It is confounded (the "strong proposer" was an
  author-added vocabulary hint) and is a re-discovery of the published verification-gap / reward-model
  overoptimization phenomenon. It is reported as a **scope limitation**, not a headline.
- **The recon-over-valuation "structural limit" is NOT a headline contribution either.** We show three
  *post-hoc* fixes fail, but we do **not** prove the failure is unavoidable by alternative training/simulator
  design, nor that other LLM-pentest systems hit the same wall. As evidenced it cannot survive the standard
  "is it avoidable / is it general?" attack, so it is reported as an **honest limitation + diagnosis**, not a
  third contribution.

---

## Main scientific question (the spine)

> **Can a per-step process-reward evaluator for penetration testing be obtained cheaply — learned in an
> abstract single-host-web simulator with ZERO real-target labels and NO leakage — and transferred to steer
> an LLM agent through complete, real end-to-end web kill chains to root?**

The spine is a **label-free, transferable process-reward model for pentest, validated on real whole-machine
kill chains.** Proposer/prompt quality is not the spine; the recon-bias limit is analysis, not the spine.

---

## Positioning vs related work (why this is novel)

A 2026-06 survey found **no prior work pairs a distilled/learned vertical pentest model with a process-reward
model, and none transfers a PRM from an abstract simulator to real targets.** Closest systems take a
different route:
- **Pentest-R1** — 8B LLM, two-stage RL, **no PRM** (outcome/sparse signal).
- **xOffense** — Qwen3-32B LoRA fine-tune, **no PRM, no transfer**.
- **HackMentor** — instruction-tuned security assistant, not an autonomous process-reward agent.
- **PentestEval / AutoPenBench** — outcome-level benchmarks; no process-level (per-step) pentest reward.

So our three contributions occupy empty ground: the **process-reward formulation for pentest** (C1), the
**label-free simulator-to-real acquisition** of it (C2), and the **real whole-machine end-to-end validation**
of the transferred signal (C3).

---

## Contributions

### ★ C1 — A process-reward evaluator for penetration testing (formulation + model)
We lift autonomous pentest from a **sparse, terminal** success signal to a **dense, per-step process reward**:
a model (the **Pentest-PRM**) that scores how much each candidate action advances the kill chain, and uses
that score to re-rank the agent's next move. To our knowledge this is the **first process-reward model for
penetration testing** (competitors use RL on outcome signals, no PRM).
- *Evidence:* the per-step process gain is real and clustered-significant — **prm 51.7 % vs llm_only 34.3 %
  per-step progress, +17 pp, episode-clustered p = 0.0001** (16 web boxes); the PRM is a calibrated ranker (deployed pairwise
  0.89 all / 0.98 new-instance / 0.80 new-chain; oracle top-3 0.94, rank-corr +0.46).
- *Scope:* a process *improver/steerer*, not an exploit generator — the exploit class is assumed known and
  expressible in the frozen 16-action schema + per-target η-recipe.

### ★ C2 — Label-free acquisition: RL value-oracle in an abstract simulator → distilled PRM → gated transfer
The expensive part of any PRM is the **per-step labels**. We obtain them for free: train an **RL (DQN) value
oracle** in an abstract single-host-web MDP, use its values to **label** action quality, **distil** a
state-conditioned PRM from those labels, then transfer it to real targets through a **gated φ/ψ/η adapter**
(φ: real output→abstract state; ψ: LLM text→16 frozen actions; η: action→concrete command) — with **zero
real-target labels** and **no leakage of hidden task ground truth**.
- *Evidence:* φ/ψ/η + `safety.py` gate; **schema coverage ~92 %** (out-of-abstraction 8.5 %); **ψ normalizer
  95.5 %** on a labeled benchmark / **78.5 %** on hard held-out fixtures (from a 49 % baseline); a leakage
  audit confirms the PRM input carries no secret path/credential/flag and degrades gracefully under masking.
- *Novel delta:* not "a PRM" but a **PRM whose step-labels are produced by a label-free abstract RL oracle and
  transferred via a safety-gated text↔schema adapter** — the route none of Pentest-R1 / xOffense take.

### ★ C3 — Real whole-machine end-to-end validation + a value-localization finding
The transferred process reward **autonomously drives the complete real kill chain — Web entry → foothold →
same-host privilege escalation → root** — on real VulnHub VMs, and we **localize where its value comes from.**
- *Evidence:* **DC-1 pooled n = 18: prm reaches root 100 % (18/18) vs llm_only 56 % (10/18)**;
  a **phase-split** shows the raw LLM **collapses to 9 % per-step progress in the privilege-escalation phase**
  while the advised agent sustains 37 % — i.e. **the process reward's value is concentrated in the hardest,
  local-privesc phase**, which only a whole-machine target exercises. 16 Docker web boxes add breadth; Toppo
  is a deterministic full-chain confirmation.
- *Robustness (not model-specific):* a symmetric **3-vendor** study (DeepSeek / Qwen-3.7-max / GPT-5.4)
  reproduces the behavior — Joomla rescue on all three, top-1 ranking prm > llm_only on 20/21 vendor-boxes.
- *Novel delta:* the **first demonstration that a simulator-transferred process-reward signal steers an agent
  through a full real kill chain to root**, with the value empirically attributed to the privesc phase.

### Supporting — Leak-free, cluster-robust evaluation
Episode-clustered permutation tests + cluster bootstrap + CRN-paired reranker-isolation ablation + Holm
multiple-comparison correction. Shows naive two-proportion tests overstate significance. **Supporting rigor,
not a headline contribution.**

---

## Honest limitations (reported in full, NOT headline contributions)

### L1 — Structural recon over-valuation under sim→real transfer (diagnosis, scope-limited)
The transferred value **over-values reconnaissance** (`web_path_enum` mean label **0.89** ≫ `exploit` **0.54**
≫ `command_execution` / `privilege_escalation` **0.04**), because the masked abstract training rarely produces
"recon-when-already-known" states. We frame this as an instance of **reward-model overoptimization under
covariate shift induced by training-time action masking** (cf. Gao et al. 2023) and provide a reproducible
diagnostic. **Honest scope — what we do and do NOT claim:** we show **three independent *post-hoc* fixes**
(inference reweight / label-correction retrain / top-k restriction) **fail**; we do **NOT** claim the failure
is unavoidable by alternative training/simulator design, nor that other LLM-pentest systems provably hit it.
It is therefore a **characterized limitation of *this* transfer recipe**, reported for honesty and as a
caution, **not** advanced as a general "boundary" contribution.

### L2 — Proposer-conditional reranker value
The reranker's *outcome* benefit depends on the proposer's own ordering. It helps a weak/un-prompt-engineered
proposer (per-step +17 pp, p = 0.0001), but once the proposer is coached with an explicit action-vocabulary
hint it improves on its own (its goal-rate rises 0.16 → 0.53, wasted-step rate 0.52 → 0.32) and, on an
isolated ranking-only test, the PRM's per-step progress (0.27) is **no better than a random re-ordering of the
same candidates (0.29)**. We report this efficiency inversion in full and **decline to center the paper on
it**, because (a) it is confounded with prompt engineering and (b) it is an instance of the known
verification-gap. (Per-episode terminal benefit on single-service web boxes is likewise often tied — stated
plainly; the per-step vs per-episode distinction is kept explicit throughout.)

---

## Honesty guardrails (must hold)

1. Per-step (process) claims and per-episode (outcome) claims are **separated**; the efficiency inversion
   (llm_only ≥ PRM under a coached proposer) is **reported in full** as L2, never hidden.
2. Box count = **16 Docker web (single-host single-service) + 2 VM full-chain (whole-machine: DC-1, Toppo)**.
   No inflated count. **Raven-2 omitted** (CVE-2016-10033 foothold blocked by image-hardened PHPMailer).
   **Symfonos-1** = dropped boundary. **XBEN/XBOW dropped (provenance-only)**, replaced by the scope statement.
3. Leakage wall: PRM input = observable context only; no oracle q-values, no hidden task ground truth;
   walkthroughs / creds / payloads stay in η plumbing, never in proposer/PRM/φ context.
4. 16-action schema stays frozen.
5. Every quantitative claim traces to a report under `outputs/` (the Experiments chapter is fully
   claim-checked; see `PAPER_EXPERIMENTS_{EN,ZH}.md`).

---

## Old framing → current role (so nothing is lost / confused)

| Old label | Old role | Current role |
|---|---|---|
| C1 (φ/ψ/η transfer) | system | **part of C2 (the transfer method)** |
| value-transfer "spine" | spine | **demoted — now the *method* under C2, not the headline** |
| process-reward / PRM | implicit | **C1 — the headline object (formulation + model)** |
| full-chain VMs | evidence | **C3 — real end-to-end validation + value-localization** |
| recon-bias structural gap | "C-C spine" | **L1 — honest limitation/diagnosis (scope-limited)** |
| proposer-conditional crossover | "core finding" | **L2 — honest limitation** |
| CRN + clustered methodology | contribution | **Supporting rigor** |
| proposer-confidence gate | proposed | **Future work (optional)** |

---

## Evidence status (all core experiments COMPLETE, 2026-06-22)

| Contribution | Status |
|---|---|
| C1 process-reward evaluator | ✅ verified (per-step +17 pp p=0.0001, 16 boxes; pairwise 0.89/0.98/0.80) |
| C2 label-free acquisition + transfer | ✅ verified (coverage 92 %, ψ 95.5 %/78.5 %, leak audit) |
| C3 real whole-machine end-to-end | ✅ verified (DC-1 100 % vs 56 %, phase-split, Toppo; 3-vendor robustness) |
| Supporting methodology | ✅ verified |
| L1 recon over-valuation | ✅ verified (label evidence + 3 failed post-hoc fixes) — reported as limitation |
| L2 proposer-conditional | ✅ verified — reported as limitation |

---

## Future work (optional, not required for the paper)
- Proposer-confidence gate (decide *when* to trust the reranker) — builds on L2.
- Testing whether L1's recon over-valuation is avoidable by an alternative training/simulator design, or is
  inherent to label-free abstract transfer — the experiment that would promote L1 toward a contribution.
- Exploit-discovery beyond the frozen schema (currently out of scope).
