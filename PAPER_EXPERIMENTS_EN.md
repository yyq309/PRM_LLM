# Experiments

We organize the evaluation around the paper's spine — *can a process-reward / value prior, learned
cheaply in an abstract single-host-web simulator with **zero** real-target labels and **no** leakage, be
transferred through a gated adapter to steer an LLM agent through real end-to-end web kill chains, and what
are the capabilities and structural limits of that transfer?* Accordingly the experiments answer four
questions: **(i)** does the abstract simulator yield a value prior that ranks genuine progress (§E.2);
**(ii)** does that prior transfer to real targets and add per-step process value (§E.3); **(iii)** does it
drive *complete* real kill chains to root (§E.4); and **(iv)** what is its structural limit (§E.5). §E.6
shows the behavior is not specific to a single LLM, and §E.7 reports the honest limitation.

## E.1 Experimental setup

### E.1.1 Stage 1 — abstract single-host-web simulator

The value prior is trained entirely in an abstract MDP with a **frozen 16-action schema**; no real target
is touched in Stage 1. The benchmark is **65 de-templated, family-tagged tasks** spanning **12 chain
topologies** (difficulty: 10 easy / 15 medium / 40 hard; chain depth 2–12). The split is **45 train / 20
held-out**, where the held-out set is deliberately split into **10 unseen-instance** and **10 unseen-chain**
tasks; the two unseen-chain families (`rce_privesc`, `leak_login`) are *whole topologies absent from
training*, and the chain-signature overlap with the training set is **0**. Every task ships an `expert_plan`
and length-matched `expert_trajectory`.

We adopt a **hard mode** (tight budget = `plan_len + 2`) so that same-state decisions are *consequential*;
this de-saturates the metrics (e.g. masked goal-rate drops from a saturated 1.0 to 0.70, and the oracle's
value-return correlation rises from ≈0 to +0.38 — see §E.2). The training pipeline is
`task_config → transitions → DQN value oracle → PRM labels → Pentest-PRM`. Crucially, the PRM consumes
**only observable state+action features**; oracle q-values are **never** a PRM input feature.

### E.1.2 Stage 2 — gated abstract→real adapter

Stage 2 runs an autonomous loop `propose → ψ → PRM-rerank → η → execute → φ` against **real targets**:
- **φ** maps real tool output to the abstract observation;
- **ψ** maps free-form LLM action text to one of the 16 frozen actions;
- **η** maps an abstract action to a concrete, per-target command;
- a **safety gate** (`safety.py`) allow-lists every command (host/port/transport scoped, audited).

The **leakage wall** is strict: the PRM and proposer see observable context only — never oracle q-values,
hidden task ground truth, walkthroughs, credentials, or payloads (those live exclusively in the η plumbing).

**Targets.** **15 Docker web boxes** (single-host single-service real CVEs — ThinkPHP, Struts2, Joomla,
php-cgi, Drupalgeddon2, etc., bound to `127.0.0.1`) for breadth, plus **2 VulnHub VMs** (whole-machine
multi-step chains: **DC-1**, **Toppo:1**) on an isolated host-only segment for the complete
*Web → foothold → same-host privilege-escalation → root* chain. (Raven-2 was omitted — its CVE-2016-10033
foothold is blocked by an image-hardened PHPMailer; XBEN/XBOW is repo-provenance only, not a paper result.)

### E.1.3 A/B protocol and statistical methodology

Every real-target result is an **A/B** between two arms that share the *same* LLM proposer and differ only
in action selection: **`prm`** (the PRM reranks the proposer's candidate actions) vs **`llm_only`** (the
proposer's own ordering). Metrics: terminal `root_flag_captured` / goal-rate (non-gameable), per-step
progress rate, and a **top-1 ranking accuracy** (how often the chosen action matches the oracle-priority
pick). Because rollouts are episode-structured and branch, we use **episode-clustered permutation tests**,
**cluster-bootstrap CIs**, and a **reranker-isolation ablation** with **common random numbers (CRN)** —
the two arms are paired on coincident states. We report **effect sizes** (risk difference / ratio, odds
ratio, Cohen's *h*) alongside *p*-values and apply **Holm–Bonferroni** correction to the confirmatory
family. Naive two-proportion tests overstate significance; we report the cluster-robust numbers.

## E.2 Stage 1: a value prior that ranks genuine progress (foundation for C-A)

**RL value oracle.** Under the hard, de-saturated benchmark the oracle's genuine signal is an **expert
top-1 lift of +0.093–0.127 over random-within-mask**, a **masked goal-rate of 0.70** (vs 0.54 random) and
a **permissive (mask-free) goal-rate of 0.40** (vs 0.013 random, ≈30×). Most importantly, **where decisions
are outcome-relevant**, the oracle value correlates with realized return at **Spearman +0.378** (up from
≈0 under loose budgets) and the decision-relevant fraction rises to 0.49 (2.4×). The oracle is *weak but
genuinely informative exactly where it matters*.

**Q\* check.** Against a value-iteration **goal-aligned Q\*** (reward only on the goal-reaching transition),
the oracle ranks progress correctly: **top-1 agreement 0.74, top-3 hit 1.00, mean Spearman +0.45**. The
negative correlation against *literal* Q\* is a **decoy-milking artifact** (literal Q\* milks distractor
`path_found` rewards before terminating), not a real oracle weakness.

**Pentest-PRM ranking quality** (strong gradient-boosted model, oracle-labeled subset). The PRM that
consumes the oracle labels is a strong per-step ranker:

| Split | Pairwise | Rank | Note |
|---|---|---|---|
| oracle_all | **0.942** | 0.799 | structured state+action features (baseline TF-IDF only 0.884) |
| unseen-instance | **0.980** | 0.972 | new instances of seen chains |
| unseen-chain (hard) | **0.920** | 0.617 | whole topologies absent from training |

We report the **oracle-labeled subset** only; the full-set pairwise (0.93) is inflated because 73 % of
held-out rows are rule-labeled `score=0` and trivially self-predictable. Post-calibration more than halves
the hardest-slice ECE (unseen-chain 0.155 → 0.067). **Error-action identification** (does the PRM flag
actions that should *not* be taken?) reaches **ROC-AUC 0.927 / PR-AUC 0.887** on the genuine
valid-but-low-value subset, with perfect recall on hard-constraint errors (precondition-missing / unsafe /
out-of-scope / schema-gap / ambiguous = 1.00).

**Generalization and integrity.** Zero-shot transfer to **structurally new chain topologies** holds
(**pairwise 0.858**, rank lift +0.32 over floor, vs 0.942 in-distribution) — the PRM keys on state
features, not chain identity. Multi-seed (3×80k) oracle labels agree at **rank-agreement 0.97**. A
**leakage audit** finds **no hidden-truth token** in PRM input and **0 cliff fields** under masking — the
PRM relies on observable context, not leaked secrets. Finally, in a **closed-loop policy eval** the PRM is
*not* a standalone policy (permissive goal-rate 0.00, mask-saturated otherwise): it is a per-step **reward
model / ranker**, and its value is the ranking quality above — which autonomous rollout does not measure.

## E.3 Stage 2: abstract→real value transfer (C-A)

The gated adapter makes the transfer real. **Schema coverage is 92 %** and the **ψ normalizer improves
from 49 % → 78.5 %** accuracy on held-out real LLM action text. On the 15 Docker web boxes, the PRM
reranking the proposer's candidates yields a **statistically significant per-step process improvement**:
pooled, episode-clustered permutation gives **p = 0.02** (Holm-adjusted **0.04** on the confirmatory
family), i.e. the PRM beats random reranking of the same candidate set per step. This is the load-bearing
C-A evidence: a prior learned with zero real labels, transferred through the gate, measurably improves
real per-step action selection. (We separate *per-step* from *per-episode*: terminal goal benefit on the
web-only boxes is limited/often tied — stated plainly and revisited in §E.4 and §E.7.)

## E.4 Stage 2: real end-to-end kill chains (C-B, flagship)

The flagship evidence is the **complete** chain — *Web entry → foothold → same-host privilege escalation →
root* — on real VMs, scored by the non-gameable **`root_flag_captured`**.

**DC-1 (autonomous, deepseek-chat).** Pooling two identically-configured runs (n=10 + n=8 = **n=18**), the
**`prm` arm reaches root in 18/18 (100 %)** episodes vs **`llm_only` 10/18 (56 %)** — a **+44 pp** rescue.
The `prm` arm also uses ≈2× fewer steps. An honest note on variance: the raw proposer's autonomous root-rate
on DC-1 swings between runs (40 % at n=10, 75 % at n=8), while the PRM is **100 % in both**; we therefore
report the pooled 56 % rather than a single favorable draw, and frame the PRM's DC-1 value as **reliability**.
A **phase-split** explains *why* the PRM helps here whereas it was efficiency-neutral on the web-only boxes:
the `prm` arm makes progress in **both** phases (web ≈36 % / local ≈37 %), but **`llm_only` collapses in the
LOCAL / privilege-escalation phase (9 %)** while still doing web recon (32 %). **The PRM's full-chain value
lives in the local/privesc phase, where the raw proposer's own ordering is weakest** — exactly the regime a
web-only A/B cannot exercise.

**Toppo:1 and the proposer ceiling.** Both arms score 0 % autonomously (the proposer never builds the
required `credential-discovery → ssh` foothold), yet the **deterministic** proposer reaches root on both
DC-1 and Toppo — so the **adapter (φ/ψ/η) is sound**; the failure is a *proposer-capability ceiling*
(`exploit_never_proposed`), not an adapter defect. This cleanly separates "the value prior can't steer what
the proposer never proposes" from "the transfer is broken."

## E.5 Structural limit: recon over-valuation (C-C)

The transferred value has a **characterizable, surgically un-fixable** limitation. Because the *masked*
abstract training rarely produces "recon-when-the-answer-is-already-known" states, the PRM **over-values
recon**: the mean PRM label for `web_path_enum` is **0.887**, far above `exploit` at **0.535**. On real
targets this manifests as the PRM adding exploratory steps a competent proposer does not need. We attempted
**three independent fixes — inference-time reweighting, label-correction retraining, and top-k restriction —
and all three failed** to remove the bias without destroying ranking quality elsewhere. A multi-seed
analysis (seeds 0–4) further shows the recon-advanced-label fraction is **seed-dependent** (mean ≈0.25, std
≈0.06) and that the deployed checkpoint's 0.609 is a high outlier (a seed-gate selection artifact), so the
bias magnitude is itself noisy. We frame this as **reward-model overoptimization under covariate shift
induced by action *masking* during training** — a generalizable warning for sim-to-real value transfer,
distinct from RLHF policy drift.

## E.6 Cross-vendor generalization (multi-LLM)

To show the transfer behavior is **not specific to a single LLM**, we ran a **symmetric 3-vendor** A/B —
the *same* current code on the *same* 7 boxes (DC-1 full chain + 6 web boxes), llm proposer, prm vs
llm_only, 0 errored — across **DeepSeek-chat** (official endpoint), **Qwen-3.7-max**, and **GPT-5.4** (the
latter two via an OpenAI-compatible gateway). Headline claims were independently verified against the raw
logs. One **proposer-conditional mechanism** explains all three vendors.

**Outcome rescue (proposer-conditional).** On **Joomla CVE-2017-8917**, the PRM rescues a struggling
proposer's goal-rate for **all three vendors**: DeepSeek **1.0 vs 0.4**, Qwen **1.0 vs 0.4**, GPT-5.4
**0.6 vs 0.0** (each n=5; the load-bearing evidence is the *consistency of direction across three vendors*,
not any single small-n CI). On **DC-1**, the rescue is **conditional on proposer weakness**: DeepSeek
(weaker proposer) is rescued (pooled root **100 % vs 56 %**), whereas Qwen and GPT already saturate the box
unaided (100 % = 100 %, no outcome headroom for the PRM).

**Ranking lift (near-universal).** The PRM's per-decision **top-1 ranking accuracy beats the raw proposer
on 20 / 21 vendor-boxes**, across all three vendors. The single exception (DeepSeek-Joomla, 0.264 < 0.403)
is a **metric artifact** — on that same box the PRM still wins the actual *goal* 100 % vs 40 %; top-1 is
scored against the oracle's heuristic priority, which is not goal-truth on every box.

**Per-step is box-dependent, not uniform.** The PRM improves per-step progress on the longer **multi-step
chains** (DC-1, php-cgi) but **hurts** it on the short **single-shot** Struts2 boxes, where the raw proposer
already fires the one correct action every step and the PRM's exploration only dilutes the rate. The same
pattern appears for all three vendors.

Together: outcome rescue is *conditional* on the proposer being weak; ranking lift is *near-universal*;
the per-step effect tracks chain length — one re-ranking mechanism, different ceilings, **not
DeepSeek-specific.**

## E.7 Honest limitation: proposer-conditional reranker value

We report — and decline to center the paper on — the limitation that the reranker's **per-episode** benefit
depends on the proposer's own ordering quality. It helps a weak / un-prompt-engineered proposer (per-step
**+12 pp**, p = 0.02), but a proposer given an explicit action-vocabulary hint can make the reranker
**net-negative** (`llm_only` 66.7 % vs `prm` 39.6 %, p < 0.0001 on that configuration). We treat this as a
**scope limitation** rather than a headline because (a) it is confounded with prompt engineering — the
"strong" proposer there was an author-added vocab hint, not a natural variable — and (b) it is an instance
of the known verification-gap / reward-model-overoptimization phenomenon. The efficiency inversion is
reported in full, never hidden to protect the spine.

## E.8 Summary

The abstract simulator yields a **weak-but-genuine** value prior that ranks real progress (Spearman +0.45
vs goal-aligned Q\*; PRM pairwise 0.94 / unseen-chain 0.92), with **no leakage** and **zero real labels**
(§E.2). The gated adapter transfers it to real targets with a **significant per-step process improvement**
(p = 0.02; §E.3), and drives **complete real kill chains to root** where the PRM's value concentrates in the
privilege-escalation phase (DC-1 pooled 100 % vs 56 %; §E.4). The transfer has a **characterized structural
limit** — recon over-valuation that resists three fixes (§E.5) — and the whole behavior **replicates across
three LLM vendors** under one proposer-conditional mechanism (§E.6). The reranker's per-episode value is
**conditional** on proposer weakness, reported honestly as a limitation (§E.7).
