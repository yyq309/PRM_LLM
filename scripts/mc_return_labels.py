"""Monte-Carlo return labels from real WebAttackSim rollouts (auxiliary/calibration labels).

For each oracle-labeled PRM sample (o_t, a), instead of trusting only the DQN's Q-estimate, we
RESTORE the env to o_t (by replaying the task's expert trajectory to that step and deep-copying the
state), FORCE the candidate action a once, then continue with a COMPETENT rollout policy
(masked-greedy DQN — reaches the goal ~1.0, unlike the weak maskless policy) until done/budget, and
accumulate the discounted return G_t. We compute G under TWO reward models:
  - full   : the env reward (discovery + goal + penalties) — same shaping as the DQN labels.
  - goal   : +1 only on the goal-reaching transition (gamma^steps-to-goal) — non-degenerate, no
             decoy-milking (the degeneracy we confirmed for literal Q*).

The env is DETERMINISTIC and masked-greedy is deterministic, so K=1 rollout is exact here (variance
would require a stochastic rollout policy — noted, not used in this prototype).

Per same-state candidate group we then derive mc_gap = max_b G(o_t,b) - G(o_t,a) and compare the
DQN's value ordering to the realized MC ordering (agreement on the best action, Spearman, gap
magnitude, recovery rate). This quantifies how trustworthy the DQN labels are and produces an
MC label that can relabel/calibrate them. MC = V^pi (under masked-greedy), NOT V* — documented.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import DQNValueOracle  # noqa: E402
from generate_prm_dataset import select_checkpoint  # noqa: E402
from train_prm_baseline import read_jsonl  # noqa: E402
from web_attack_sim import Action, ActionType, WebAttackSimEnv, normalize_llm_action  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

GAMMA = 0.98


def action_from_norm(na: dict[str, Any]) -> Action | None:
    at = na.get("action_type")
    if not at:
        return None
    return Action(ActionType(at), target=na.get("target"), parameter=na.get("parameter"))


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if spearmanr is not None:
        rho = spearmanr(a, b).correlation
        return None if rho != rho else float(rho)
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ar, br)[0, 1])


def mc_rollout(env: WebAttackSimEnv, oracle: DQNValueOracle, state, forced: Action, max_steps: int) -> dict[str, Any]:
    """Force `forced` from `state`, then masked-greedy DQN rollout. Return G_full, G_goal, reached."""
    env.state = copy.deepcopy(state)
    obs, reward, done, _t, _i = env.step(forced)
    g_full = float(reward)
    g_goal = 1.0 if (env.state and env.state.done and env._goal_reached()) else 0.0
    discount = GAMMA
    steps = 0
    while not done and steps < max_steps:
        vec = env.encode_observation(obs)
        q = oracle.q_values(vec)
        mask = env.action_mask(permissive=False)
        allowed = [i for i, m in enumerate(mask) if m]
        if not allowed:
            break
        a = max(allowed, key=lambda i: q[i])
        obs, reward, done, _t, _i = env.step(a)
        g_full += discount * float(reward)
        if env.state and env._goal_reached():
            g_goal += discount * 1.0
            break
        discount *= GAMMA
        steps += 1
    reached = bool(env.state and env.state.done and env._goal_reached())
    return {"g_full": g_full, "g_goal": g_goal, "reached": reached}


def snapshot_expert_states(task_path: Path) -> tuple[dict[int, Any], int]:
    """Replay the expert trajectory, snapshotting the state BEFORE each step."""
    task = load_task_config(task_path)
    env = WebAttackSimEnv()
    obs, _info = env.reset(task_path)
    snaps: dict[int, Any] = {}
    for t, raw in enumerate(task.get("expert_trajectory", [])):
        snaps[t] = copy.deepcopy(env.state)
        n = normalize_llm_action(raw)
        if n.action is None:
            break
        obs, _r, done, _t, _i = env.step(n.action)
        if done:
            break
    return snaps, env.max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo return labels from masked-greedy rollouts (auxiliary/calibration).")
    parser.add_argument("--inputs", nargs="+", default=["outputs/prm_samples_train.jsonl", "outputs/prm_samples_heldout.jsonl"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed-gate-report", type=Path, default=ROOT / "outputs" / "oracle_seed_gate.json")
    parser.add_argument("--allow-ungated-oracle", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--labels-output", type=Path, default=ROOT / "outputs" / "mc_return_labels.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "mc_return_report.json")
    parser.add_argument("--limit-groups", type=int, default=0, help="Smoke cap on candidate groups (0 = all).")
    args = parser.parse_args()

    checkpoint, _gate = select_checkpoint(args)
    oracle = DQNValueOracle(checkpoint, args.device)

    rows: list[dict[str, Any]] = []
    for inp in args.inputs:
        p = (ROOT / inp) if not Path(inp).is_absolute() else Path(inp)
        for r in read_jsonl(p):
            if r.get("label_source") == "oracle":
                rows.append(r)

    # Group oracle samples by (task_path, step).
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("task_path")), int(r.get("step", 0)))].append(r)

    snap_cache: dict[str, tuple[dict[int, Any], int]] = {}
    env = WebAttackSimEnv()
    labeled: list[dict[str, Any]] = []
    group_keys = list(groups)
    if args.limit_groups:
        group_keys = group_keys[: args.limit_groups]

    for gi, (task_path, step) in enumerate(group_keys):
        if task_path not in snap_cache:
            snap_cache[task_path] = snapshot_expert_states(Path(task_path))
        snaps, max_steps = snap_cache[task_path]
        state = snaps.get(step)
        if state is None:
            continue
        members = groups[(task_path, step)]
        recs = []
        for r in members:
            action = action_from_norm(r.get("normalized_action") or {})
            if action is None:
                continue
            mc = mc_rollout(env, oracle, state, action, max_steps)
            rec = {
                "sample_id": r.get("sample_id"),
                "candidate_group": r.get("candidate_group"),
                "task_path": task_path,
                "step": step,
                "dataset_split": r.get("dataset_split"),
                "dqn_score": float(r.get("score", 0.0)),
                "dqn_value_gap": r.get("value_gap"),
                "mc_g_full": round(mc["g_full"], 4),
                "mc_g_goal": round(mc["g_goal"], 6),
                "mc_reached_goal": mc["reached"],
            }
            recs.append(rec)
        if not recs:
            continue
        # Per-group MC gap (goal-aligned) and best.
        best_goal = max(rr["mc_g_goal"] for rr in recs)
        for rr in recs:
            rr["mc_gap_goal"] = round(best_goal - rr["mc_g_goal"], 6)
        labeled.extend(recs)

    # ---- aggregate DQN-vs-MC agreement ----
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rr in labeled:
        by_group[rr["candidate_group"]].append(rr)

    MEANINGFUL = 0.02  # a same-state group is "decision-relevant" only if realized returns actually differ
    best_agree = 0
    n_multi = 0
    within_spearmans = []
    gap_goal_vals = []
    # restricted to groups whose realized MC return actually varies (where ranking is meaningful)
    n_meaningful = 0
    best_agree_meaningful = 0
    spearmans_meaningful = []
    for g, recs in by_group.items():
        if len(recs) < 2:
            continue
        n_multi += 1
        goal_vals = [rr["mc_g_goal"] for rr in recs]
        dqn_best = max(range(len(recs)), key=lambda i: recs[i]["dqn_score"])
        if recs[dqn_best]["mc_g_goal"] >= max(goal_vals) - 1e-9:
            best_agree += 1  # DQN's pick is (tied-)optimal under MC
        rho = spearman([rr["dqn_score"] for rr in recs], goal_vals)
        if rho is not None:
            within_spearmans.append(rho)
        gap_goal_vals.extend(rr["mc_gap_goal"] for rr in recs)
        if (max(goal_vals) - min(goal_vals)) > MEANINGFUL:
            n_meaningful += 1
            if recs[dqn_best]["mc_g_goal"] >= max(goal_vals) - 1e-9:
                best_agree_meaningful += 1
            if rho is not None:
                spearmans_meaningful.append(rho)

    reached = [int(rr["mc_reached_goal"]) for rr in labeled]
    report = {
        "checkpoint": str(checkpoint),
        "rollout_policy": "force action a, then masked-greedy DQN to done/budget (competent; deterministic K=1)",
        "reward_models": "g_full (env reward) and g_goal (+1 at goal only, gamma^steps-to-goal)",
        "n_oracle_samples": len(labeled),
        "n_candidate_groups_multi": n_multi,
        "dqn_pick_is_mc_optimal_rate": round(best_agree / max(n_multi, 1), 4),
        "within_group_mean_spearman_dqn_vs_mcgoal": round(float(np.mean(within_spearmans)), 4) if within_spearmans else None,
        "frac_groups_positive_spearman": round(float(np.mean([int(s > 0) for s in within_spearmans])), 4) if within_spearmans else None,
        "mean_mc_gap_goal": round(float(np.mean(gap_goal_vals)), 4) if gap_goal_vals else None,
        "median_mc_gap_goal": round(float(np.median(gap_goal_vals)), 4) if gap_goal_vals else None,
        "forced_action_goal_recovery_rate": round(float(np.mean(reached)), 4) if reached else None,
        "meaningful_gap_threshold": MEANINGFUL,
        "frac_groups_decision_relevant": round(n_meaningful / max(n_multi, 1), 4),
        "n_groups_decision_relevant": n_meaningful,
        "dqn_pick_is_mc_optimal_rate_DECISION_RELEVANT": round(best_agree_meaningful / max(n_meaningful, 1), 4) if n_meaningful else None,
        "within_group_mean_spearman_DECISION_RELEVANT": round(float(np.mean(spearmans_meaningful)), 4) if spearmans_meaningful else None,
        "note": (
            "KEY FINDING: forced_action_goal_recovery_rate ~1.0 and median_mc_gap_goal ~0 => under a competent "
            "(masked-greedy) continuation the env is FORGIVING: almost any single action is recoverable, so the "
            "realized value-gap between same-state candidates is ~0 on most groups. The 'weak' DQN/PRM ranking is "
            "therefore largely NOISE ON A FLAT LANDSCAPE, not a labeling error. The decision-relevant numbers "
            "(groups whose realized returns actually differ, > meaningful_gap_threshold) are the honest test of "
            "ranking quality; the all-group averages are diluted by ties. MC is V^pi(masked-greedy), not V*."
        ),
    }
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with args.labels_output.open("w", encoding="utf-8") as f:
        for rr in labeled:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"oracle samples MC-labeled: {len(labeled)}  multi-candidate groups: {n_multi}")
    print(f"DQN pick is MC-optimal rate: {report['dqn_pick_is_mc_optimal_rate']}")
    print(f"within-group Spearman(DQN score, MC goal-return): {report['within_group_mean_spearman_dqn_vs_mcgoal']} "
          f"(frac positive {report['frac_groups_positive_spearman']})")
    print(f"mean / median MC goal-gap: {report['mean_mc_gap_goal']} / {report['median_mc_gap_goal']}")
    print(f"forced-action goal recovery rate (env forgiveness): {report['forced_action_goal_recovery_rate']}")
    print(f"decision-relevant groups (returns actually differ): {report['frac_groups_decision_relevant']} "
          f"({report['n_groups_decision_relevant']}/{n_multi})")
    print(f"  DQN pick MC-optimal on DECISION-RELEVANT groups: {report['dqn_pick_is_mc_optimal_rate_DECISION_RELEVANT']}")
    print(f"  Spearman(DQN,MC) on DECISION-RELEVANT groups: {report['within_group_mean_spearman_DECISION_RELEVANT']}")


if __name__ == "__main__":
    main()
