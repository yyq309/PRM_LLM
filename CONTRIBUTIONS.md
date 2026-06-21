# Contributions & Paper Main-Line (converged 2026-06-21)

This is the **canonical framing** for the paper. It supersedes the earlier "proposer-conditional
value / proposer-competence crossover"-centric framing. The decision (2026-06-21): **do NOT build
the paper on prompt/proposer quality.** That axis (a) walks into the reviewers' strongest, hardest-
to-defend attack (the "strong proposer" was an author-added vocab hint — confounds competence vs
candidate-surface coverage), (b) is a re-discovery of the published verification-gap / reward-model
overoptimization phenomenon, and (c) shifts the spine toward prompt engineering, away from what the
project actually built. So the conditionality is **demoted to an honest limitation**, not the headline.

---

## Main scientific question (the spine)

> **Can a process-reward / value prior, learned cheaply in an abstract single-host-web simulator with
> ZERO real-target labels and NO leakage, be transferred via a gated adapter to steer an LLM agent
> through real end-to-end web kill chains — and what are the capabilities and structural limits of
> that transfer?**

The spine is **abstract→real value transfer + its mechanistic limits + real end-to-end validation.**
Proposer/prompt quality is NOT the spine.

---

## Contributions (re-ordered to the new spine)

### ★ C-A (spine) — Abstract→real value-transfer adapter
A two-stage method: train a value/process-reward prior in an abstract single-host-web MDP (Stage-1
DQN oracle → Pentest-PRM), then transfer it via a **gated φ/ψ/η adapter** (φ: real output→abstract
state; ψ: LLM text→16 frozen actions; η: action→concrete command) to drive an LLM agent on **real
targets** — with **zero real-target labels** and **no leakage of hidden task ground truth**.
- *Evidence:* φ/ψ/η + `safety.py` gate; schema coverage 92%; ψ normalizer 49%→78.5%.
- *Novel delta (position vs SGFT / sim-to-real value shaping):* the security domain, the text↔schema
  mapping with a safety gate, and the **structurally masked abstract MDP** as the value source.

### ★ C-B (spine, flagship evidence) — Real end-to-end kill-chain validation
The transferred value + adapter drives an autonomous agent through **complete real kill chains**:
**Web entry → foothold → same-host privilege escalation → root flag** on real VMs (the full-chain
VulnHub experiment, see `STAGE2_FULLCHAIN_PLAN.md`), plus 15 Docker web boxes for breadth.
- *Evidence:* `root_flag_captured` (non-gameable terminal metric) on the VM boxes; per-step process
  value of the PRM-as-reranker on the web boxes (per-step pooled clustered p=0.02).
- *Honest scope:* per-step process improver ✓; per-episode terminal goal benefit is limited / often
  tied — reported honestly (see C-Limitation). **Scope (design assumption):** the exploit class is assumed
  KNOWN and expressible within the 16-action schema + per-target η-recipe; **autonomous discovery of novel
  multi-step exploits is out of scope** (the PRM is a value/ranking model, not an exploit generator). XBEN/XBOW
  is NOT a paper result — it is provenance-only in the repo (`stage2_xben_autonomous.json`, 0/18 confirmed the
  boundary); the paper states the scope as an assumption instead.

### ★ C-C (spine, mechanism + rigorous negative) — Structural train-inference gap & its limits
The transferred value has a **characterizable structural limitation**: the **masked** abstract training
rarely produces "recon-when-already-known" states, so the PRM **over-values recon**
(`web_path_enum` label 0.887 ≫ `exploit` 0.535) and extrapolates it to real targets. This is
**surgically un-fixable** — three independent fixes (inference reweight / label-correction retrain /
top-k restriction) all failed.
- *Position vs literature:* frame as reward-model overoptimization under **covariate shift induced by
  action MASKING during training** (cite Gao et al. 2210.10760), distinct from RLHF policy drift.
- *Value:* a generalizable warning for sim-to-real value transfer.

### Supporting — Leak-free evaluation + cluster-robust methodology
Episode-clustered permutation tests + cluster bootstrap + reranker-isolation ablation (proposer-fixed +
CRN-paired). Shows naive two-proportion tests overstate. **Kept as supporting rigor, not a headline
contribution** (CRN + cluster-robust inference are standard; the only novel sliver is the
partial-pairing-under-divergence analysis for branching agent rollouts).

---

## Honest limitation (DEMOTED, reported — NOT hidden, NOT headline)

**Proposer-conditional reranker value.** The PRM-as-reranker's benefit depends on the proposer's own
ordering quality: it helps a weak/un-prompt-engineered proposer (per-step +12pp, p=0.02) and a
well-prompted proposer can make it net-negative (with a vocab hint: llm_only 66.7% vs prm 39.6%,
p<0.0001). We **report this as a scope limitation** — "a well-prompted proposer can obviate the
reranker" — and explicitly **decline to center the paper on it**, because (a) it is confounded with
prompt engineering (the strong proposer was an author-added vocab hint, not a natural variable) and
(b) it is an instance of the known verification-gap (cite 2509.17995 / 2506.18203). **This finding is
reported in full in the limitations section; the efficiency inversion is not buried.**

---

## Honesty guardrails (must hold)

1. The efficiency inversion (llm_only ≥ PRM under a good proposer) is **reported in full**, as a
   limitation — never hidden to protect the spine.
2. Box count = **15 Docker web (single-host single-service) + 2 VM full-chain (whole-machine multi-step:
   DC-1, Toppo)**. No inflated count. **Raven-2 omitted** (CVE-2016-10033 foothold blocked by image-hardened
   PHPMailer; not load-bearing — would only have been a 3rd deterministic privesc vector). **Symfonos-1** =
   dropped boundary. **XBEN/XBOW dropped (provenance-only)**, replaced by the scope statement above.
3. Per-step vs per-episode claims are **separated**: per-step process improver ✓; terminal goal
   benefit limited/tied — stated plainly.
4. Leakage wall: PRM input = observable context only; no oracle q-values, no hidden task ground truth;
   walkthroughs/creds/payloads stay in η plumbing, never in proposer/PRM/φ context.
5. 16-action schema stays frozen.

---

## Old framing → new role (so nothing is lost / confused)

| Old label | Old role | New role |
|---|---|---|
| C1 (φ/ψ/η transfer) | system | **C-A — spine** |
| C2 (proposer-conditional crossover) | "core finding" | **DEMOTED → honest limitation** (cite verification-gap) |
| C3 (CRN + clustered methodology) | contribution | **Supporting rigor** (not headline) |
| C4 (recon-bias distribution gap) | mechanism | **C-C — spine** |
| C5 (proposer-confidence gate) | proposed | **Future work** (built on the demoted conditionality; optional) |
| (new emphasis) full-chain VMs | evidence | **C-B — spine flagship evidence** |

---

## Evidence status

| Contribution | Status |
|---|---|
| C-A adapter transfer | ✅ verified (15 web boxes, coverage/ψ numbers) |
| C-B end-to-end kill chain | ⏳ planned (full-chain VMs, `STAGE2_FULLCHAIN_PLAN.md`) + ✅ per-step web result |
| C-C structural gap / limits | ✅ verified (label evidence + 3 failed fixes) |
| Supporting methodology | ✅ verified |
| Limitation (conditionality) | ✅ verified (reported, demoted) |
| Future: proposer gate, multi-vendor LLM | ⏳ optional |

---

## Implications for ongoing experiments

- **Full-chain VM experiment** = the flagship C-B evidence → its priority is **raised**, plan unchanged
  (`STAGE2_FULLCHAIN_PLAN.md`).
- **Different-vendor LLM** = re-scoped from "prove the crossover generalizes" to "show transfer behavior
  is not deepseek-specific" → still useful, **lower stakes / can be deferred**.
- **2×2 / continuous proposer-quality sweep / interaction test** (the old P0 defense of the crossover) =
  **no longer on the critical path** — the conditionality is now a reported limitation, not a claim to
  defend.
- **Proposer-confidence gate (old C5)** = optional future work.
