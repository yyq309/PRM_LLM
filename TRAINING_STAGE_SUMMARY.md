# WebAttackSim — Training-Stage Summary (Stage 1)

Paper/report-grade closure of the **training stage only** (Web-RL value oracle →
Pentest-PRM label transfer). **No** real-target adapter, Docker/VulnHub, deployment, or
inference is included — those are Stage 2 and explicitly out of scope here.

Machine-readable counterpart: [`outputs/training_stage_summary.json`](outputs/training_stage_summary.json)
(assembled directly from the report files by `scripts/build_training_summary.py`, so it
cannot drift from the artifacts).

The label-transfer chain is `task_config → transitions → RL value oracle → PRM labels →
Pentest-PRM`. The PRM consumes **only observable state+action features** — oracle q-values
are **never** a PRM input feature.

> **CANONICAL = HARD MODE (env improvement #1 adopted).** `tasks/` now use tight budgets
> (`plan_len + 2`, `hard_mode=true`) so same-state decisions are **consequential**. This
> **de-saturates** the metrics: masked goal is now **0.70** (vs random 0.54), not the old
> 1.0=1.0; and the oracle's value correlates with realized return **where decisions matter**
> (Spearman **−0.005 → +0.378**). The numbers below are the HARD canonical;
> `outputs/training_stage_summary.json` is the authoritative current-numbers source.
> (`generate_tasks.py` is hard by default; `--loose` reproduces the legacy budgets.)

---

## 1. Task set & split (main experiment)

| | value |
|---|---|
| tasks | **65** de-templated, family-tagged |
| topology families | **12** (12 distinct expert_plan signatures) |
| difficulty | easy 10 / medium 15 / hard 40 |
| chain depth | 2 – 12 steps |
| split | **45 train / 20 held-out** (10 unseen-instance + 10 unseen-chain) |
| unseen-chain families | `rce_privesc`, `leak_login` (whole topologies absent from train) |
| chain-signature overlap with train | **0** |
| every task has `expert_plan` + `expert_trajectory` | ✅ (lengths matched, all 65) |

Families: `leak_file`, `default_pw`, `leak_login`, `injection_login` (sqli/lfi),
`rce_shell`, `rce_privesc`, `upload_{default,leak}_{shell,privesc}`, and the three
structurally-novel coverage-extension families `authed_injection`, `chained_exploit`,
`leak_authed_privesc`. (12 original hand-authored templates were retired to
`tasks_archive/` and are **not** part of this experiment.)

## 2. Headline metrics (canonical oracle = `seed_gate_12fam_80k/seed_2`, permissive 80k)

**Read the honest signal, not the saturated raw numbers.**

### RL value oracle (honest_eval, HARD canonical)
| metric | value | note |
|---|---|---|
| **expert top-1 lift over random-within-mask** | **+0.127** | ✅ the genuine oracle signal (was +0.093 loose) |
| expert top-1 | 0.463 | |
| masked goal | **0.700** (random 0.544) | ✅ **de-saturated** — a real +0.16 gap now (was 1.0=1.0) |
| permissive (maskless) goal | 0.400 (random 0.013) | ≈30× random |
| **Spearman(value, realized return) where decisions matter** | **+0.378** | ✅ was −0.005 (≈random) under loose; the core win |
| decision-relevant group fraction | **0.49** | was 0.20 loose (2.4×) |

### Q\* value-iteration check (verify_qstar, all 65 tasks) — literal vs goal-aligned
The "weak ranking" against literal Q\* is a **decoy-milking artifact**, not a real oracle weakness.
Literal-reward Q\* milks distractor `path_found` rewards before terminating; **goal-aligned Q\***
(reward = +1 only on the goal-reaching transition, so milking strictly delays the goal and can never
be preferred) is the sensible optimum. Against it the oracle ranks progress correctly:

| metric (non-degenerate, ≥4 allowed) | literal-Q\* | **goal-aligned Q\*** |
|---|---|---|
| top-1 agreement | 0.13 | **0.74** |
| top-3 hit | 0.79 | **1.00** |
| mean Spearman | **−0.41** | **+0.45** |
| per-decision gap fraction | 0.61 | 0.40 |

The oracle **does** rank genuine progress-toward-goal correctly (top-1 0.74, Spearman +0.45); the
negative literal-Q\* Spearman was the milking artifact. The strict gate flag is still `False`
(per-decision gap 0.40 > 0.25), and the headline oracle signal remains the **+0.093 top-1 lift over
random** under the mask — but the "oracle ranking is weak" reading is corrected. Reproduce both:
`verify_qstar.py` (literal) and `verify_qstar.py --reward-mode goal`.

### Pentest-PRM — strong state-conditioned model (oracle-labeled subset only, HARD canonical)
| split | pairwise | rank | ECE |
|---|---|---|---|
| oracle_all | **0.890** | 0.799 | 0.166 |
| unseen-instance | **0.980** | 0.972 | 0.032 |
| **unseen-chain** (hard) | 0.800 | 0.617 | 0.347 |

Honest tradeoff: hard mode is a **harder, de-saturated benchmark** (decisions are consequential),
so oracle_all / unseen-chain pairwise are lower than the old loose 0.94 / 0.92 — the problem is now
non-trivial, not a regression. unseen-instance actually rose (0.964 → **0.980**). The error-action
identification ROC-AUC is 0.89 (was 0.93 loose); fork-prevention (score) rose 0.42 → **0.50**.

⚠️ The **full-set** PRM pairwise (0.93) is **inflated**: 73% of held-out rows are rule-labeled
with score=0 and are trivially self-predictable. Only the oracle-labeled subset is reported as
the headline.

### Error-action identification (error-trajectory scoring, `error_action_eval.py`)
The env reward design grades errors with negative rewards (−0.1 step-cost … −3.0 unsafe/invalid);
this measures whether the PRM **flags actions that should NOT be taken**. Binary error = `rank_label
∉ {high, medium}`.

| cut | error ROC-AUC | PR-AUC | rank-head P/R/F1 |
|---|---|---|---|
| full-set (rule-inflated) | 0.992 | 0.998 | 0.99 / 0.99 / 0.99 |
| **oracle subset (genuine: valid-but-`low`)** | **0.927** | **0.887** | **0.91 / 0.92 / 0.91** |

Per-error-category recall: `precondition_missing` / `unsafe` / `outside_scope` / `schema_gap` /
`ambiguous` = **1.00**; the subtle valid-but-low-value action = **0.92** (flagged) / 0.85 (exact
category). So the PRM identifies hard-constraint errors perfectly and valid-but-wrong moves at
0.92 recall / 0.91 precision.

### Trajectory-level failure credit assignment (`trajectory_credit_eval.py`)
Synthetic derailment: at a mid-trajectory fork (k=n//2) inject a premature goal-grab that skips the
real step and breaks the chain (derail_failure_rate **0.75**). Two findings:
- **Per-step credit assignment is weakly positive**: per-step PRM score vs realized Monte-Carlo
  return-to-go Spearman **0.28** (75% of trajectories positive); the derail step is scored **0.24**
  lower than the preceding correct steps.
- **Prospective root-cause prevention at the fork is WEAK**: the PRM prefers the correct action over
  the premature goal-grab only **0.42** (score) / **0.25** (rank-head). The PRM recognizes a bad step
  after the fact but is **not a reliable fork-gate** — a limitation inherited from the weak oracle
  (masked top-1 lift +0.093).

### MC-return labels: diagnosis + validated fix (`mc_return_labels.py`, `mc_relabel_eval.py`)
Real-rollout Monte-Carlo labels: restore each oracle state (expert replay + deep-copy), force the
candidate, then continue with a competent **masked-greedy** rollout to compute the realized discounted
return G (full and goal-aligned reward). The env is DETERMINISTIC, so K=1 is exact (variance would need
a stochastic rollout policy).
- **Diagnosis**: forced-action goal recovery **0.95** → the env is forgiving, so only **20% (100/495)**
  of same-state decisions are outcome-relevant. On those, the DQN value has **~0 correlation with
  realized return (Spearman −0.005)** and picks the realized-best only **0.46**. ⇒ the headline 0.94
  pairwise is dominated by trivial/flat groups; **the oracle labels are uninformative exactly where
  decisions matter**.
- **Productionized fix (honest after bootstrap CIs)**: a prototype MC-only relabel *looked* large
  (0.29→0.68) but that used oracle-only training. Under the canonical all-rows setup with bootstrap
  CIs, the decision-relevant top-1 gain (0.29→0.44) is **directional, not significant** (only 34
  groups; CIs overlap), and **pure MC overfits** (tanks the DQN-headline 0.94→0.58 and the global
  pairwise). The robust, **adopted** result is a conservative blend **y = 0.3·Q + 0.7·MC** that lifts
  the global **pairwise-vs-realized-MC from 0.473 → 0.522** (+0.049, over all oracle pairs) at a ~0.05
  headline cost. Persisted as `prm_strong_mcblend.joblib` (α=0.3).
- **Net**: MC labels give a **real but modest** improvement to ranking-vs-realized-truth. The deeper
  point — that the env is forgiving so few decisions matter, and the PRM input features (not only the
  label source) limit fork discrimination — is the honest ceiling. MC = V^π(masked-greedy), not V*.

### Four PRM variants (all on the same canonical seed-2 labels, oracle-labeled subset)
| variant | oracle_all pairwise | unseen-chain pairwise | note |
|---|---|---|---|
| baseline (TF-IDF + ridge) | 0.884 | — | honest_eval baseline |
| robust (dirty-obs + conf-weighting) | 0.934 (clean) | — | ECE 0.035 (cal +0.041 vs baseline); **OOD held-out family B**: rank-drop −0.002 vs baseline +0.005 (slightly better), pairwise −0.017 (slightly worse) → marginal/mixed OOD benefit |
| joint MLP (`L_gap+L_rank+L_diag+L_pref`) | 0.943 | 0.925 | `L_pref` redundant on dense value_gap labels (ablation, reported) |
| **strong (gradient boosting)** | **0.942** | **0.920** | **headline** |

Structured state+action features (baseline 0.884 → strong/joint 0.94) add real signal. Multi-seed
oracle label confidence (3 × 80k seeds): seed-rank agreement **0.97**, full-agreement **0.92**.

**Post-calibration (ECE, sigmoid/Platt):** the hard unseen-chain ECE is more than halved
(**0.155 → 0.067**) and oracle_all improves (0.101 → 0.075), but calibration **degrades** the
already-well-calibrated unseen-instance (0.049 → 0.149) — so it is a net win only on the hard slice
and should be applied selectively. The persisted `prm_strong.joblib` carries the sigmoid-calibrated
rank head. **Closed loop:** wiring the strong PRM into `evaluate_prm_policy` gives **no lift** over the
TF-IDF baseline (permissive goal 0.00 for both) — the PRM is a step ranker, not an autonomous policy;
its value is the per-decision ranking quality above, which autonomous rollout does not measure.

### Leakage masking audit (leakage_audit, 65-task / seed_2)
No hidden-truth leak (no path/credential/flag token in PRM input) and masking each observable
context field degrades metrics only **gracefully** (0 cliff fields) → the PRM relies on observable
context, not leaked secrets. Oracle q-values are **not** a PRM feature.

### Zero-shot transfer to structurally-new topologies (eval_new_family_zeroshot)
PRM trained on original families only (new-family rows excluded, same canonical oracle labels):
**pairwise 0.858** (rank lift **+0.32** over floor) vs in-distribution 0.942. Per family:
authed_injection 0.888, chained_exploit 0.826, leak_authed_privesc 0.858. → new chain topologies
within the abstraction largely transfer because the PRM keys on state features, not chain identity.

### Multi-seed label confidence (held-out, 3 × 80k oracle seeds)
3-seed rank agreement **0.97**, full-agreement **0.92**, mean multiseed confidence 0.61.

### Learning-curve saturation (family_learning_curve)
Unseen-chain pairwise plateaus by ~K=4–5 families; adding more same-abstraction families/instances
sharpens rank/calibration, not a new capability ceiling. ⇒ within the 16-action abstraction,
**task count / topology count is not the binding constraint** — the real ceiling is the abstraction
(≈49% of real DeepSeek actions are out-of-schema), which is Stage 2.

### Closed-loop policy eval (evaluate_prm_policy, 20 held-out)
| policy | masked+guard goal | permissive (no mask/guard) goal |
|---|---|---|
| expert | 1.00 | — |
| oracle | 0.55 | 0.10 |
| prm (TF-IDF baseline) | 0.55 | 0.00 |
| random_valid | 0.55 | 0.00 |

⚠️ Masked closed loop is **mask-saturated** (oracle = prm = random_valid). Permissive: value models
are weak as **standalone policies**. The PRM is a process **reward model (ranker)**, not a policy;
its value is the ranking quality above, and the closed-loop path here uses the weak TF-IDF baseline,
not the strong gradient-boosted PRM.

### Env improvement #1 — consequential decisions (VALIDATED, opt-in)
The MC analysis showed the env is *forgiving* (forced-action recovery ~0.95) so only ~20% of
same-state decisions affect the outcome — that scarcity is what caps the oracle/PRM. A **tight-budget
hard mode** (`generate_tasks.py --hard` → `tasks_hard/`, budget = plan_len + 2; hard oracle in
`outputs/hard/`) makes decisions consequential. Retraining the oracle on it (80k×3):

| metric (where decisions matter) | loose canonical | **hard** |
|---|---|---|
| forced-action goal recovery | 0.95 | **0.70** |
| decision-relevant group fraction | 0.20 | **0.49** (2.4×) |
| oracle picks realized-best | 0.46 | **0.57** |
| **Spearman(oracle value, realized return)** | **−0.005** (≈random) | **+0.373** |

So consequential decisions turn the oracle from *uninformative where it matters* into *genuinely
informative* — directly lifting the ceiling that limits the PRM. **Backward compatible / default OFF**:
the loose canonical pipeline is untouched; fully adopting hard mode is a deliberate next-iteration
retrain. (`outputs/hard/hard_vs_loose.json`.)

## 3. Honest caveats (do not omit in any report)

1. Masked goal_rate / top-3 are saturated; report **top-1 lift over random** (+0.093).
2. verify_qstar literal-Q\* strict gate **fails**, but that is a **decoy-milking artifact**: against goal-aligned Q\* the oracle ranks progress correctly (top-1 0.74, Spearman +0.45). The remaining honest limit is the modest masked top-1 lift (+0.093), not Q\*-rank inconsistency.
3. PRM full-set metrics inflated by rule rows (73% score=0) — report **oracle-subset** only.
4. Closed-loop goal_rate is mask-saturated; PRM is a ranker, not a policy.
5. Zero-shot 0.858 < in-distribution 0.942; unseen-chain ECE 0.155 (calibration worse on hardest split).
6. RL oracle is weak / mask-dependent (maskless goal 0.35); needed 80k steps over 45 tasks to avoid the 0.05 undertraining regression — **scale training-steps with the train-task count**.
7. Real-world ceiling is the **abstraction**, not task count — a Stage-2 concern, not addressed here.

## 4. Reproducing (Stage 1, in `WebAttackSim/`)

**One command** (core chain, slow steps gated):
```powershell
python .\scripts\run_training_stage.py                 # smoke->coverage->labels->PRMs->evals->summary->tests
python .\scripts\run_training_stage.py --include-slow   # + oracle retrain, verify_qstar, MC rollouts, permissive policy
```
It prints a PASS/FAIL manifest and exits non-zero on any failure. Or run the steps individually:

```powershell
# Oracle (canonical, ~minutes/seed on CPU; pins seed_2 via canonical_checkpoint):
python .\scripts\run_oracle_seed_gate.py --seeds 0 1 2 --training-steps 80000 --train-no-action-mask --output-dir .\outputs\seed_gate_12fam_80k --aggregate-output .\outputs\oracle_seed_gate.json --min-goal-rate 0.7 --min-expert-top3-rate 0.6 --max-expert-avg-gap 6.0

# Label transfer + all evaluations (bare commands -> canonical seed_2):
python .\scripts\generate_prm_dataset.py        # PRM labels (train 45 / heldout 20)
python .\scripts\train_prm_baseline.py          # TF-IDF baseline PRM
python .\scripts\train_prm_strong.py            # strong state-conditioned PRM (headline)
python .\scripts\honest_eval.py                 # honest oracle/PRM numbers + caveats
python .\scripts\verify_qstar.py                # Q* value-iteration label check
python .\scripts\coverage_audit.py              # 65-task / 12-family diversity grid
python .\scripts\family_learning_curve.py       # data-scaling saturation
python .\scripts\eval_new_family_zeroshot.py    # zero-shot structural transfer
python .\scripts\evaluate_prm_policy.py         # closed-loop (heldout_all 20)
python .\scripts\build_training_summary.py      # assemble this summary's JSON
python -m pytest tests\ -m "not slow"           # 303 passed / 1 skipped
```

The canonical oracle is pinned by `outputs/oracle_seed_gate.json → canonical_checkpoint`
(honored by both `select_checkpoint` and `resolve_gated_checkpoint`), so every bare command
uses the same seed-2 oracle. The split is structural and deterministic.

## 5. Stage 2 (NOT done — out of scope)

Docker/VulnHub adapter (φ/ψ/η), real-target out-of-abstraction rate, real A/B uplift. The
Stage-1 evidence says the binding constraint for real-world effect is the action-schema
coverage and the per-step realism gap — both Stage-2 concerns.
