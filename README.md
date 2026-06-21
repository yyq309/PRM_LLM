# WebAttackSim

WebAttackSim is a lightweight single-host Web penetration-testing simulator for
training a Web-RL / HAPT value oracle. It is intentionally abstract: no real HTTP
requests are executed during RL training. Real Web labs such as DVWA, WebGoat,
Juice Shop, VulnHub, HTB-style targets, or AutoPenBench-style containers should
be connected later through an adapter.

> **Resuming Stage 2 (real-VulnHub inference)? Start with [`STAGE2_HANDOFF.md`](STAGE2_HANDOFF.md)** —
> a complete onboarding doc: architecture, the 12 live targets, every experiment + honest finding,
> exact reproduce commands, conventions, gotchas, and the ranked next steps. Canonical results live in
> [`STAGE2_LIVE_RESULTS.md`](STAGE2_LIVE_RESULTS.md).
>
> For a local VMware full-VM lab that adds host-level privilege-escalation coverage, see
> [`stage2/VMWARE_VULNHUB_LAB.md`](stage2/VMWARE_VULNHUB_LAB.md) and
> [`stage2/vm_target_registry.json`](stage2/vm_target_registry.json).

## Design

```text
task_config -> env.reset() -> observation
action      -> env.step()  -> next_observation + reward + feedback + done
```

The task configuration is hidden ground truth. The RL policy only receives a
minimal Plan-B entry observation: target HTTP service and `/` are known, while
hidden paths, parameters, credentials, vulnerabilities, shells, and flags must
be discovered through abstract Web actions.

## Files

```text
WebAttackSim/
  tasks/                    65 de-templated task configs (12 families)
  web_attack_sim/           simulator package
  scripts/smoke_test_env.py quick environment check
  scripts/demo_pipeline.py  stage-one demo: normalizer + oracle labels
```

## Quick Check

From this directory:

```powershell
python .\scripts\smoke_test_env.py
```

Expected result: all bundled sample tasks can be solved by scripted action
sequences, and the script prints the final observation, reward, and trace.

Export random transitions for RL pipeline debugging:

```powershell
python .\scripts\export_random_transitions.py --episodes 50 --output .\outputs\random_transitions.jsonl
```

Each JSONL row contains `obs`, `obs_vec`, `action_id`, `action_type`, `reward`,
`next_obs`, `next_obs_vec`, `done`, and structured `feedback`.

Run the stage-one demo pipeline:

```powershell
python .\scripts\demo_pipeline.py
```

The demo uses the bundled DQN checkpoint, normalizes LLM-style action text into
abstract Web actions, queries `Q_web`, `V_web`, and `value_gap`, then writes:

- `outputs/demo_prm_samples.jsonl`: `context + action -> value_gap/rank/diagnosis`
  samples for PRM prototyping.
- `outputs/demo_candidate_ranking.json`: 3-5 candidate actions per state ranked
  by demo process value.

## Oracle Retraining and Gate Check

Before generating formal PRM labels, retrain and validate a masked Web-RL value
oracle. The strict action mask keeps completed or precondition-missing actions
out of DQN action selection and Bellman targets, while permissive environment
mode remains available for low-value-action analysis.

```powershell
python .\scripts\train_dqn.py --training-steps 8000 --eval-episodes 5 --output .\outputs\web_dqn_masked.pt
python .\scripts\evaluate_oracle.py --checkpoint .\outputs\web_dqn_masked.pt --report-output .\outputs\oracle_eval_masked.json --q-report-output .\outputs\oracle_q_gap_masked.jsonl
```

The train/held-out split is computed **structurally** by `task_split.py` (no manual
task lists): whole topology families are held out as **unseen-chain** and one
instance per trained family as **unseen-instance**. The canonical oracle is the
permissive (maskless-trained) multi-seed gate over that split:

```powershell
python .\scripts\run_oracle_seed_gate.py --seeds 0 1 2 --training-steps 80000 --train-no-action-mask --output-dir .\outputs\seed_gate_12fam_80k --aggregate-output .\outputs\oracle_seed_gate.json --min-goal-rate 0.7 --min-expert-top3-rate 0.6 --max-expert-avg-gap 6.0
```

It trains the same split for each seed, evaluates the held-out tasks, and writes
aggregate min/mean/max/std to `outputs/oracle_seed_gate.json`, with per-seed
checkpoints and q-gap reports under `outputs/seed_gate_12fam_80k/seed_*`. (A short
masked single-checkpoint run such as `oracle_eval_masked.json` is for debugging
only.) Note: when the train-task count grows, scale `--training-steps`
proportionally — the maskless permissive goal regresses under-trained while the
masked top-k metrics still look fine (45k over 45 tasks dropped permissive goal to
0.05; 80k restored it to 0.35).

Only allowed actions in q-gap JSONL reports receive numeric `value_gap`;
disallowed normalized actions should be labeled by rules such as
`precondition_missing`, `credential_missing`, or `unsafe`.

`evaluate_oracle.py` also reports expert-plan ranking agreement:

- `expert_action_top1_rate`: how often the scripted expert action is the
  highest-valued allowed action.
- `expert_action_top3_rate`: how often the expert action is within the top 3
  allowed actions.
- `expert_action_avg_value_gap`: average gap between the state value and the
  expert action value.

**Honest reporting rule (method §10.1, `honest_eval.py`):** do NOT headline raw
`eval_goal_rate` / `expert_action_top3_rate` — a random-within-mask policy matches
them (mask saturation). Report the **expert top-1 lift over a random-within-mask
baseline** as the genuine oracle signal, and only quote goal_rate/top-3 with that
baseline beside them. Top-1 over top-3 is also preferred because several Web states
have multiple valid orderings (e.g. fingerprinting before retrieval).

The main-experiment set contains **65 de-templated tasks across 12 topology
families** — `leak_file`, `default_pw`, `leak_login`, `injection_login` (sqli/lfi),
`rce_shell`, `rce_privesc`, `upload_{default,leak}_{shell,privesc}`, and the three
structurally-novel families `authed_injection`, `chained_exploit`,
`leak_authed_privesc` — across easy/medium/hard tiers. The structural split is
**45 train / 20 held-out (10 unseen-instance + 10 unseen-chain)**. Every task
carries `expert_plan` and `expert_trajectory`, so smoke tests and PRM-sample
generation need no script-local task tables. (Twelve original hand-authored
templates were retired to `tasks_archive/` once the de-templated generator
replaced them; they are not part of the main experiment.)

Generate and (re)write the programmatic tasks with:

```powershell
python .\scripts\generate_tasks.py            # dry-run: verify all are solvable
python .\scripts\generate_tasks.py --write    # persist verified tasks to tasks\
```

The generator composes parametrized vulnerability-chain instances from a primitive
library, injects distractor / dead-end paths (so the oracle and PRM learn to avoid
`failed_branch_overcommitment`), and self-verifies every task is solvable by both
its structured `expert_plan` and its natural-language `expert_trajectory` before
writing. Audit task diversity along the method §12.1 axes (vuln class × attack
surface × chain depth × difficulty) with:

```powershell
python .\scripts\coverage_audit.py
```

This writes `outputs/coverage_audit.json` with per-cell counts and the empty cells
that remain structurally implausible under the frozen 16-action schema (candidates
for a §6.1 versioned schema extension). `tasks/VULNHUB_CORRESPONDENCE.md` grounds
each abstract family in representative real boxes / walkthroughs and OWASP/CWE
labels for the §14 adapter to later validate.

The normalizer distinguishes unsupported outputs by method-level categories:
`schema_gap` for in-scope Web actions not covered by the current 16-action
schema, `outside_single_host_web_scope` for actions outside the single-host Web
setting, `unsafe` for destructive/out-of-scope actions, and `ambiguous` for
under-specified actions.

After the seed gate passes, generate formal PRM label datasets:

```powershell
python .\scripts\generate_prm_dataset.py
```

By default, the generator selects the best passed seed-gate checkpoint and
writes:

- `outputs/prm_samples_train.jsonl`
- `outputs/prm_samples_heldout.jsonl`
- `outputs/prm_candidate_ranking.json`
- `outputs/prm_dataset_summary.json`

Each sample carries `dataset_split`, `oracle_checkpoint`, `label_version`,
normalizer/schema/oracle confidence fields, the oracle q-report when available,
and a diagnosis label. The candidate pool includes expert actions, valid
lower-value actions, precondition failures, schema gaps, unsafe actions,
outside-scope actions, and ambiguous actions.

Train a lightweight TF-IDF PRM baseline for sanity checking the generated labels:

```powershell
python .\scripts\train_prm_baseline.py
```

This writes `outputs/prm_baseline.joblib`,
`outputs/prm_baseline_eval.json`, and
`outputs/prm_baseline_predictions.jsonl`. The baseline predicts the scalar
process score, `rank_label`, and diagnosis from `context + action + normalized
action schema`; it does not use oracle q-values as input features.

Evaluate the trained PRM as a closed-loop candidate reranker:

```powershell
python .\scripts\evaluate_prm_policy.py
```

This writes `outputs/prm_policy_eval.json` and compares `expert`, `oracle`,
`prm`, and `random_valid` policies on the held-out tasks. The default controller
uses schema filtering, the strict action mask, and a visible-state precondition
guard. Current held-out result: `prm` reaches `goal_rate=1.0` with no failed
steps.

For a noisier ablation without the strict action mask:

```powershell
python .\scripts\evaluate_prm_policy.py --policies oracle prm random_valid --no-action-mask --report-output .\outputs\prm_policy_eval_unmasked.json
python .\scripts\evaluate_prm_policy.py --policies prm random_valid --no-action-mask --no-precondition-guard --report-output .\outputs\prm_policy_eval_unmasked_no_guard.json
```

With the visible-state guard enabled, `prm` still reaches `goal_rate=1.0` on the
held-out tasks. Without both the mask and guard, the PRM policy collapses to
`goal_rate=0.0`, which confirms that the PRM should be deployed with the method's
schema/action-validity guard rather than as an unconstrained executor.

## Label-Credibility Checks

Two checks make the oracle labels auditable rather than taken on faith. They are
the method's two label-credibility gates (method §5.1/§12.1 and §10.1).

### Q* value-iteration verification

Each task is a deterministic MDP whose hidden ground truth is fully known,
and under the integer-action policy space the remaining budget strictly decreases
each step, so the reachable state graph is a finite DAG. This lets us compute the
exact optimal action-value `Q*(o,a)` by backward induction (no approximation) and
verify that the trained DQN oracle's ranking is consistent with it — proving the
labels are not the random product of a single RL run.

```powershell
python .\scripts\verify_qstar.py --report-output .\outputs\qstar_report.json
```

The report compares the oracle against exact `Q*` over every reachable state. The
method's §5.1 ask is *ranking consistency* (the labels are not the random product
of a single RL run), so the primary gate is `dqn_qstar_top3_hit_rate` plus a
scale-relative `oracle_greedy_value_gap_fraction` (the oracle greedy's value-gap as
a fraction of mean `V*`, since absolute gaps scale with reward magnitude). Exact and
decisive top-1 agreement are diagnostics. Against **literal-reward** `Q*` the 65-task
non-degenerate (≥4-allowed) top-3 hit is **0.79**, per-decision gap fraction **0.61**,
and mean Spearman **−0.41** — which would read as "weak ranking". But this is a
**decoy-milking artifact**: literal-`Q*` enumerates distractor `+path_found` rewards
before terminating, so its "optimal" ranking is degenerate. Re-running against
**goal-aligned `Q*`** (`--reward-mode goal`: reward = +1 only on the goal-reaching
transition, so milking strictly delays the goal and can never be optimal) shows the
oracle ranks genuine progress **correctly**: non-degenerate top-1 **0.74**, top-3
**1.00**, Spearman **+0.45**. So the negative literal-Spearman was the artifact, not a
real oracle weakness. The strict gate flag stays `False` (per-decision gap 0.40 > 0.25),
and the headline oracle signal remains the small expert top-1 lift over random
(+0.093, see the honest-reporting rule above) — but the "oracle ranking is weak"
reading is corrected. Run both modes:

```powershell
python .\scripts\verify_qstar.py --report-output .\outputs\qstar_report.json                       # literal
python .\scripts\verify_qstar.py --reward-mode goal --report-output .\outputs\qstar_report_goal.json # goal-aligned
```

The check also surfaced a reward-shaping subtlety it reports explicitly: on short
tasks that reach a terminal reward while distractor paths exist, the literal-reward
`Q*` optimum first enumerates the decoy paths (`+path_found` each) before grabbing
the flag, because reaching the goal ends the episode — inflating `V*` and the
absolute gap on the easy leak tasks. The DQN oracle does **not** milk decoys (it
reads the flag promptly), i.e. it is *more* sensible than literal-`Q*` there, which
is exactly why the gate uses ranking consistency rather than absolute `Q*`
optimality.

This check originally surfaced a reward bug in `privilege_escalation`, which
re-granted the `+8` `privilege_escalated` reward on every call even at the target
privilege; the unmasked `Q*` optimum exploited it as a reward-milk loop. The
handler now returns `duplicate_action` when already at the target privilege,
consistent with every other handler. (Strict-mask training never took the action,
so masked oracle/seed-gate/PRM results are unchanged.)

### Leakage masking audit

The `value_gap` labels come from an oracle that knows the hidden truth, but the
PRM input must be a strict function of observable state and history. The audit has
a structural hidden-truth leak test (flag values, non-dictionary passwords,
internal file paths, and not-yet-discovered hidden paths must never appear in the
verbalized context) and a masking ablation (re-train with each observable context
field masked and check the held-out drop is graceful, not a cliff).

```powershell
python .\scripts\leakage_audit.py --report-output .\outputs\leakage_audit.json
```

Current result: zero structural leaks on both splits and graceful degradation for
every field, with the metadata-hygiene warning cleared. The verbalized context is
prefixed with an opaque scenario id (`scenario_<hash>` from
`opaque_scenario_id(task_id)`) rather than the descriptive `task_id`, so the PRM
cannot read the vulnerability family off the label; held-out scenarios hash to
ids unseen in training. Retraining the baseline after this change leaves the
held-out metrics unchanged, confirming the PRM never relied on the label.

## Training-stage Robustness

These checks harden the two trained models (value oracle and PRM) so they execute
reliably and stay good evaluators — the training-stage robustness the method makes
a release prerequisite (§5.1, §9.2, §11.2, §11.3). They do not touch the inference
/ adapter stage.

### Reward-sensitivity ablation (RL §5.1)

The reward magnitudes are hand-tuned constants, so the oracle's usefulness must rest
on relative ranking, not absolute Q. This retrains the oracle under rescaled / jittered
rewards and checks ranking is scale-invariant while value_gap scales with the reward.

```powershell
python .\scripts\reward_sensitivity.py --scales 0.5 1.0 2.0 --jitter 0.25 --seeds 0 1
```

Result: held-out goal rate stays 1.0 and expert top-3 stays ~0.87–0.92 across scales,
while expert avg value_gap scales ~linearly (0.74 → 1.29 → 3.21). Ranking is
scale-invariant; absolute Q is not (`outputs/reward_sensitivity.json`).

### Information-value alignment (RL §9.2)

Rational reconnaissance that returns nothing now but unlocks a high-value chain later
must not be undervalued. Using the distractor paths, this checks the oracle gives
*productive* enumeration a small value_gap and *dead-end* enumeration a larger one.

```powershell
python .\scripts\info_value_check.py
```

Result: productive enumeration value_gap ≈ 0.11 vs dead-end ≈ 1.29 (separation ≈ 1.18,
pairwise 93%); the oracle learned long-term path value, not surface reward
(`outputs/info_value_check.json`).

### Robust PRM training + calibration (PRM §6.1, §11.2, §11.3)

Hardens the PRM against adapter-style dirty observations and normalizer imperfection,
and reports the calibration / ranking metrics the method gates on.

```powershell
python .\scripts\train_prm_robust.py
```

It trains a baseline (clean, unweighted) and a robust model (dirty context+action
augmentation, with confidence weighting applied to ORACLE labels only — rule-based
negatives are certain and kept at full weight), then compares them on clean and dirty
held-out splits. Result: the robust model matches clean rank accuracy (~0.88) at lower
calibration error (ECE 0.132 → 0.079) and more stable dirty pairwise ranking
(0.914 → 0.926), with an abstention benefit ≈ 0.09. Writes `outputs/prm_robust.joblib`
and `outputs/prm_robust_eval.json`.

## Method-completion (training stage)

These close the remaining method.docx training-stage requirements.

- **Multi-seed oracle label confidence (§11.2)** — `oracle_label_confidence.py` regenerates
  labels from all 3 oracle seeds, and per oracle label emits seed agreement, value-gap std,
  and a `multiseed_confidence` (used as a PRM sample weight) with robust cross-seed labels.
  Result: ~80% of oracle labels have full 3-seed rank agreement.
- **Joint PRM with `L_pref` (§10)** — `train_prm_joint.py` is a torch MLP trained with the
  full joint objective `L_gap + L_rank + L_diag + L_pref` (the preference term over same-state
  candidate pairs was previously missing), confidence-weighted. Ablation: on these dense
  `value_gap` labels `L_pref` is redundant (score regression already orders within-state), so
  it is reported but not load-bearing here.
- **Q\* direct labels for simple templates (§12.1)** — `qstar_labels.py` emits exact Q\*
  value-gap labels on clean distractor-free templates. The trained DQN agrees with exact Q\*
  on only ~46% of rank buckets there, so on simple templates the Q\* labels should replace the
  DQN labels, narrowing the DQN to partial-observability / unseen-instance generalization.
- **Normalizer accuracy benchmark (§10.1)** — `normalizer_benchmark.py` reports normalizer
  accuracy on a labeled set (0.97 status / 0.95 type) plus the out-of-schema rate on real
  DeepSeek actions (~0.49). It surfaced and we fixed a real bug: short keywords (`rce`, `lfi`,
  `sqli`) matched as substrings inside `source` / `force`, mislabeling actions — now word-bounded.
- **Curriculum training (§12.1)** — `train_dqn.py --curriculum` (easy→medium→hard). Evaluated
  and **not adopted**: it did not improve the permissive goal rate on this hard-task-heavy set.

## Coverage extension: structurally-novel chain families (§13.1)

A data-scaling diagnostic (`family_learning_curve.py` → `prm_learning_curve.json`) first showed
that adding more *instances* of the existing topologies does nothing — unseen-chain pairwise
saturates by ~4 families. The binding constraint is topology *coverage*, not task count. So three
genuinely-new chain families were added, built from the **same frozen 16 primitives** but exploiting
the env's vuln `requires` preconditions (`auth_state` / `vulnerability_verified`) that no original
family used, giving expert-plan signatures unseen in the original set:

- `authed_injection` — weak login → **auth-gated** injection → shell (real: authenticated RCE).
- `chained_exploit` — two exploit stages where v2 `requires: vulnerability_verified:v1` (real: chained SQLi→RCE).
- `leak_authed_privesc` — leak creds → `credential_use` → authed RCE → privesc → root (12-step full chain).

Now **12 families / 65 tasks**. Two honest findings:

- **Zero-shot structural transfer is strong** (`eval_new_family_zeroshot.py`): a PRM trained on the
  *original* families only (new-family rows excluded; labels from the same canonical oracle) ranks the
  three new topologies at oracle-subset pairwise **0.858** (rank lift **+0.32** over the majority floor),
  vs the in-distribution **0.942** — well above the 0.5 pairwise floor. The PRM keys on observable *state
  features*, not memorized chain identity, so new topologies within the abstraction largely transfer; this
  is why the real-world gap is the **abstraction** (49% of real DeepSeek actions are out-of-schema), not
  the family count.
- **Retraining on all 12 families improves every metric** vs the 9-family baseline: oracle-subset
  pairwise 0.938→**0.942**, unseen-instance rank 0.92→**0.98**, unseen-**chain** rank 0.74→**0.87**,
  unseen-chain ECE 0.21→**0.16**, oracle top-1 0.21→**0.43**. The maskless permissive goal needed
  **80k steps** (not 45k) to avoid an undertraining regression on the larger 45-task train set
  (restored 0.05→**0.35**, ≥ the old 0.30). The unseen-chain learning curve still plateaus (~0.9),
  confirming saturation: the gains are in rank/calibration, not a new capability ceiling.

## Tests

A pytest suite verifies every module and the correctness-critical script logic,
locking in the bug fixes surfaced during development (the `privilege_escalation`
duplicate-action guard, the 34-feature encoder, the normalizer form-phrase pitfall,
Q* history-invariance, and the structural-split integrity).

```powershell
python -m pytest tests/                 # full suite (includes a tiny end-to-end DQN train)
python -m pytest tests/ -m "not slow"   # fast unit tests only
```

Coverage: action space / schemas / reward / encoder / normalizer / env dynamics,
every bundled task's solvability via both `expert_plan` and `expert_trajectory`,
the structural-family split (0 unseen-chain signature overlap), exact Q* value
iteration, leakage-audit helpers, PRM feature extraction and ranking/calibration
metrics, the DeepSeek client's response parsing, and a tiny DQN train/score
integration test. The live DeepSeek health check runs only when `DEEPSEEK_API_KEY`
is set, otherwise it is skipped.

## RL Interface

Use:

```python
from web_attack_sim import WebAttackSimEnv

env = WebAttackSimEnv()
obs, info = env.reset("tasks/backup_leak.json")
obs_vec = env.encode_observation(obs)
next_obs, reward, done, truncated, info = env.step(2)  # web_path_enumeration
```

The environment exposes:

- `env.actions`: ordered abstract action list.
- `env.action_mask(permissive=True)`: legal-action mask. The permissive mask
  keeps low-value actions available so the PRM can learn why they are bad.
- `env.encode_observation(obs)`: numeric vector for DQN-style agents.
- `info["feedback"]`: structured feedback for trace logs and PRM labels.

## First Research Use

1. Train a value oracle on the abstract tasks.
2. Export `Q_web(o,a)`, `V_web(o)`, and `value_gap`.
3. Let an LLM act in the same environment.
4. Normalize the LLM action into an abstract Web action.
5. Label the action with the value oracle.
6. Train Pentest-PRM on `context + action -> value_gap/rank/diagnosis`.
