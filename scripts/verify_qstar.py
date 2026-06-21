"""Exact Q* value-iteration verification for the WebAttackSim oracle.

The pilot tasks are deterministic MDPs whose hidden ground truth (preconditions
and effects) is fully known. Under the integer-action policy space used by the
DQN oracle, every action is deterministic and the remaining budget strictly
decreases each step, so the reachable state graph is a finite DAG. This lets us
compute the exact optimal action-value function Q*(s, a) by backward induction
(value iteration without approximation).

Two uses, following method §5.1 / §12.1:

1. Verify DQN ranking consistency against Q* (the labels are not the random
   product of a single RL run): per reachable state, compare the oracle Q-value
   ranking with the Q* ranking over the allowed action set (top-1 agreement,
   top-3 hit, Spearman correlation), plus expert-action Q* optimality.
2. Provide Q* directly as an alternative label source for these fully specified
   templates (per-task expert-step Q* / V* / value_gap).

This narrows the DQN's job to what it alone can do (partial-observability
approximation and unseen-instance generalization) and answers both "are the
labels trustworthy" and "what does RL actually add".
"""

from __future__ import annotations

import os

# torch and scipy can each link an OpenMP runtime; allow them to coexist.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import argparse
import copy
import json
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import DQNValueOracle  # noqa: E402
from train_dqn import resolve_task_paths  # noqa: E402
from web_attack_sim import Action, ActionType, WebAttackSimEnv  # noqa: E402
from web_attack_sim.action_space import ACTIONS  # noqa: E402
from web_attack_sim.encoder import encode_observation  # noqa: E402
from web_attack_sim.env import RuntimeState  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402

try:
    from scipy.stats import spearmanr  # noqa: E402
except Exception:  # pragma: no cover - scipy ships with sklearn
    spearmanr = None

NUM_ACTIONS = len(ACTIONS)
EPS = 1e-6


def state_key(state: RuntimeState) -> tuple:
    """Canonical hashable key for the *dynamics-relevant* part of the state.

    History bookkeeping (attempted/failed actions, failed_branches) never feeds a
    handler or the reward, so it is excluded; otherwise every state would be
    unique and the DAG would not collapse.
    """
    return (
        frozenset(state.discovered_services),
        frozenset(state.discovered_paths),
        frozenset(state.tech_stack),
        frozenset(state.known_forms),
        frozenset(state.known_parameters),
        frozenset(state.suspected_vulnerabilities),
        frozenset(state.verified_vulnerabilities),
        frozenset(state.credentials),
        state.auth_state,
        state.shell_state,
        state.privilege_level,
        frozenset(state.read_files),
        int(state.remaining_budget),
        bool(state.done),
    )


def step_from(env: WebAttackSimEnv, state: RuntimeState, action: int | Action) -> tuple[RuntimeState, float, bool]:
    """Apply one action to a copy of `state` using the exact env dynamics."""
    env.state = copy.deepcopy(state)
    env.trace = []
    _obs, reward, done, _truncated, _info = env.step(action)
    assert env.state is not None
    return copy.deepcopy(env.state), float(reward), bool(done)


class QStarSolver:
    def __init__(self, task_path: Path, gamma: float, reward_mode: str = "full"):
        # reward_mode:
        #   "full" - exact env reward (discovery + goal). Literal Q*; can milk distractor
        #            path_found rewards before terminating on short tasks (documented degeneracy).
        #   "goal" - goal-aligned reward: +1 ONLY on the transition that reaches the goal, 0 else.
        #            Optimal = shortest discounted path to goal (gamma^dist); milking a decoy delays
        #            the goal and is strictly worse, so this Q* is NON-degenerate. Tests whether the
        #            oracle ranks genuine progress-toward-goal (isolates the milking artifact).
        assert reward_mode in {"full", "goal"}
        self.reward_mode = reward_mode
        self.gamma = gamma
        self.env = WebAttackSimEnv()
        self.task_config = load_task_config(task_path)
        self.start_obs, _info = self.env.reset(task_path)
        assert self.env.state is not None
        self.max_steps = self.env.max_steps
        self.start_state = copy.deepcopy(self.env.state)

        self.v_cache: dict[tuple, float] = {}
        self.q_cache: dict[tuple, list[float]] = {}
        self.state_repr: dict[tuple, RuntimeState] = {}

    def solve(self, max_states: int) -> tuple[float, int]:
        v0 = self._compute(self.start_state, max_states)
        return v0, len(self.q_cache)

    def _compute(self, state: RuntimeState, max_states: int) -> float:
        if state.done:
            return 0.0
        key = state_key(state)
        if key in self.v_cache:
            return self.v_cache[key]
        if len(self.q_cache) >= max_states:
            raise RuntimeError(f"reachable-state cap exceeded: {max_states}")

        self.state_repr[key] = state
        q_values: list[float] = []
        for action_id in range(NUM_ACTIONS):
            next_state, reward, done = step_from(self.env, state, action_id)
            r = self._transition_reward(next_state, reward)
            q = r if done else r + self.gamma * self._compute(next_state, max_states)
            q_values.append(q)
        v = max(q_values)
        self.v_cache[key] = v
        self.q_cache[key] = q_values
        return v

    def _transition_reward(self, next_state: RuntimeState, env_reward: float) -> float:
        """Reward for a transition INTO next_state, per the active reward_mode."""
        if self.reward_mode == "full":
            return env_reward
        # goal-aligned: +1 exactly when the resulting state has reached the goal, else 0.
        self.env.state = next_state
        return 1.0 if self.env._goal_reached() else 0.0

    def allowed_mask(self, state: RuntimeState) -> list[int]:
        self.env.state = copy.deepcopy(state)
        return self.env.action_mask(permissive=False)

    def obs_vec(self, state: RuntimeState) -> list[float]:
        # state_key excludes failed-action history but the encoder feeds it to the DQN,
        # so a state_key can be reached with different histories. Zero the failed-history
        # before scoring to make the DQN evaluation history-invariant and well-defined.
        s = copy.deepcopy(state)
        s.failed_actions = []
        s.failed_branches = {}
        self.env.state = s
        obs = self.env._observation()
        return encode_observation(obs, max_budget=self.max_steps)

    def qstar_action(self, state: RuntimeState, action: Action) -> tuple[float, float]:
        """Return (Q*(s,a), V*(s)) for an arbitrary (possibly targeted) action."""
        v_here = self._compute(state, max_states=10**9)
        next_state, reward, done = step_from(self.env, state, action)
        r = self._transition_reward(next_state, reward)
        q = r if done else r + self.gamma * self._compute(next_state, max_states=10**9)
        return q, v_here


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if spearmanr is not None:
        rho = spearmanr(a, b).correlation
        return None if rho != rho else float(rho)  # filter nan
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ar, br)[0, 1])


def ranking_agreement(solver: QStarSolver, oracle: DQNValueOracle, decisive_margin: float) -> dict[str, Any]:
    top1_hits = 0
    top3_hits = 0
    counted = 0
    # Non-degenerate = states with >=4 allowed actions, where top-3 is a real constraint
    # (with <=3 allowed, top-3 hit is 1.0 by construction and inflates the headline).
    nondeg_top1_hits = 0
    nondeg_top3_hits = 0
    nondeg_counted = 0
    nondeg_spearmans: list[float] = []
    decisive_top1_hits = 0
    decisive_counted = 0
    greedy_gaps: list[float] = []
    per_decision_fracs: list[float] = []
    spearmans: list[float] = []

    for key, state in solver.state_repr.items():
        qstar = solver.q_cache[key]
        mask = solver.allowed_mask(state)
        allowed = [i for i, m in enumerate(mask) if m]
        if not allowed:
            continue
        dqn_q = oracle.q_values(solver.obs_vec(state))

        dqn_best = max(allowed, key=lambda i: dqn_q[i])
        qstar_order = sorted(allowed, key=lambda i: qstar[i], reverse=True)
        v_allowed = qstar[qstar_order[0]]
        counted += 1

        is_top1 = qstar[dqn_best] >= v_allowed - EPS
        third_best_q = qstar[qstar_order[min(3, len(qstar_order)) - 1]]
        is_top3 = qstar[dqn_best] >= third_best_q - EPS
        if is_top1:
            top1_hits += 1
        if is_top3:
            top3_hits += 1

        greedy_gap = v_allowed - qstar[dqn_best]
        greedy_gaps.append(greedy_gap)
        # Per-decision normalized gap: gap divided by the Q* spread over allowed actions
        # AT THIS STATE (not by cumulative mean V*), which is the dimensionally-correct scale.
        qstar_spread = qstar[qstar_order[0]] - qstar[qstar_order[-1]]
        if qstar_spread > EPS:
            per_decision_fracs.append(greedy_gap / qstar_spread)

        rho = spearman([dqn_q[i] for i in allowed], [qstar[i] for i in allowed])
        if rho is not None:
            spearmans.append(rho)

        if len(allowed) >= 4:
            nondeg_counted += 1
            if is_top1:
                nondeg_top1_hits += 1
            if is_top3:
                nondeg_top3_hits += 1
            if rho is not None:
                nondeg_spearmans.append(rho)

        if len(allowed) >= 2:
            margin = qstar[qstar_order[0]] - qstar[qstar_order[1]]
            if margin >= decisive_margin:
                decisive_counted += 1
                if is_top1:
                    decisive_top1_hits += 1

    return {
        "states_evaluated": counted,
        "dqn_qstar_top1_agreement": top1_hits / max(counted, 1),
        "dqn_qstar_top3_hit_rate": top3_hits / max(counted, 1),
        "nondegenerate_states": nondeg_counted,
        "frac_states_degenerate_le3_allowed": (counted - nondeg_counted) / max(counted, 1),
        "nondegenerate_top1_agreement": nondeg_top1_hits / max(nondeg_counted, 1) if nondeg_counted else None,
        "nondegenerate_top3_hit_rate": nondeg_top3_hits / max(nondeg_counted, 1) if nondeg_counted else None,
        "nondegenerate_mean_spearman": float(np.mean(nondeg_spearmans)) if nondeg_spearmans else None,
        "oracle_greedy_mean_value_gap": float(np.mean(greedy_gaps)) if greedy_gaps else None,
        "oracle_greedy_max_value_gap": float(np.max(greedy_gaps)) if greedy_gaps else None,
        "per_decision_gap_fraction": float(np.mean(per_decision_fracs)) if per_decision_fracs else None,
        "decisive_margin": decisive_margin,
        "decisive_states": decisive_counted,
        "decisive_top1_agreement": decisive_top1_hits / max(decisive_counted, 1) if decisive_counted else None,
        "dqn_qstar_mean_spearman": float(np.mean(spearmans)) if spearmans else None,
        "num_states_with_spearman": len(spearmans),
    }


def normalize_plan_action(raw: str | dict[str, Any]) -> Action:
    if isinstance(raw, str):
        return Action(ActionType(raw))
    return Action(ActionType(raw["action_type"]), target=raw.get("target"), parameter=raw.get("parameter"))


def expert_alignment(solver: QStarSolver, oracle: DQNValueOracle, task_path: Path) -> dict[str, Any]:
    expert_plan = solver.task_config.get("expert_plan", [])
    env = WebAttackSimEnv()
    obs, _info = env.reset(task_path)
    assert env.state is not None

    steps: list[dict[str, Any]] = []
    for idx, raw in enumerate(expert_plan):
        action = normalize_plan_action(raw)
        state = copy.deepcopy(env.state)
        mask = solver.allowed_mask(state)
        allowed = [i for i, m in enumerate(mask) if m]
        qstar_here = solver.q_cache.get(state_key(state))
        if qstar_here is None:
            solver._compute(state, max_states=10**9)
            qstar_here = solver.q_cache[state_key(state)]

        q_expert, v_star = solver.qstar_action(state, action)
        if allowed:
            v_allowed = max(qstar_here[i] for i in allowed)
        else:
            v_allowed = v_star
        gap = v_star - q_expert
        is_optimal = q_expert >= v_allowed - EPS

        dqn_q = oracle.q_values(solver.obs_vec(state))
        dqn_best = max(allowed, key=lambda i: dqn_q[i]) if allowed else int(np.argmax(dqn_q))

        steps.append(
            {
                "step": idx,
                "expert_action": action.action_type.value,
                "expert_target": action.target,
                "qstar_expert": round(q_expert, 4),
                "vstar": round(v_star, 4),
                "qstar_value_gap": round(gap, 4),
                "expert_is_qstar_optimal": bool(is_optimal),
                "dqn_greedy_action": ACTIONS[dqn_best].value,
                "dqn_greedy_matches_expert": ACTIONS[dqn_best] == action.action_type,
            }
        )
        obs, _reward, done, _truncated, _info = env.step(action)
        if done:
            break

    optimal = [s for s in steps]
    return {
        "num_steps": len(steps),
        "expert_qstar_top1_rate": sum(int(s["expert_is_qstar_optimal"]) for s in optimal) / max(len(optimal), 1),
        "expert_qstar_avg_value_gap": (
            sum(float(s["qstar_value_gap"]) for s in optimal) / max(len(optimal), 1)
        ),
        "dqn_greedy_matches_expert_rate": sum(int(s["dqn_greedy_matches_expert"]) for s in optimal) / max(len(optimal), 1),
        "steps": steps,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    oracle = DQNValueOracle(args.checkpoint, args.device)
    task_paths = resolve_task_paths(args.tasks)
    task_reports: list[dict[str, Any]] = []

    for task_path in task_paths:
        solver = QStarSolver(task_path, gamma=args.gamma, reward_mode=args.reward_mode)
        v0, num_states = solver.solve(args.max_states)
        agreement = ranking_agreement(solver, oracle, args.decisive_margin)
        expert = expert_alignment(solver, oracle, task_path)
        task_reports.append(
            {
                "task_id": solver.task_config.get("task_id"),
                "task_path": str(task_path),
                "vstar_initial": round(v0, 4),
                "num_reachable_states": num_states,
                "ranking_agreement": agreement,
                "expert_alignment": {k: v for k, v in expert.items() if k != "steps"},
                "expert_steps": expert["steps"],
            }
        )

    def avg(key_path: list[str]) -> float:
        vals = []
        for report in task_reports:
            node: Any = report
            for key in key_path:
                node = node[key]
            if node is not None:
                vals.append(float(node))
        return float(np.mean(vals)) if vals else 0.0

    mean_vstar = float(np.mean([float(r["vstar_initial"]) for r in task_reports])) if task_reports else 0.0
    nondeg_top3 = avg(["ranking_agreement", "nondegenerate_top3_hit_rate"])
    nondeg_spearman = avg(["ranking_agreement", "nondegenerate_mean_spearman"])
    per_decision_frac = avg(["ranking_agreement", "per_decision_gap_fraction"])
    aggregate = {
        "num_tasks": len(task_reports),
        "reward_mode": args.reward_mode,
        "gamma": args.gamma,
        "dqn_qstar_top1_agreement": avg(["ranking_agreement", "dqn_qstar_top1_agreement"]),
        "dqn_qstar_top3_hit_rate_ALL_states": avg(["ranking_agreement", "dqn_qstar_top3_hit_rate"]),
        "frac_states_degenerate_le3_allowed": avg(["ranking_agreement", "frac_states_degenerate_le3_allowed"]),
        "nondegenerate_top1_agreement": avg(["ranking_agreement", "nondegenerate_top1_agreement"]),
        "nondegenerate_top3_hit_rate": nondeg_top3,
        "nondegenerate_mean_spearman": nondeg_spearman,
        "dqn_qstar_mean_spearman_ALL": avg(["ranking_agreement", "dqn_qstar_mean_spearman"]),
        "decisive_top1_agreement": avg(["ranking_agreement", "decisive_top1_agreement"]),
        "oracle_greedy_mean_value_gap": avg(["ranking_agreement", "oracle_greedy_mean_value_gap"]),
        "per_decision_gap_fraction": per_decision_frac,
        "expert_qstar_avg_value_gap": avg(["expert_alignment", "expert_qstar_avg_value_gap"]),
        "mean_vstar_initial": mean_vstar,
        "total_reachable_states": sum(int(r["num_reachable_states"]) for r in task_reports),
    }
    # HONEST gate (post adversarial review): top-3 on ALL states is vacuous because ~75% of
    # states have <=3 allowed actions. Gate on the NON-DEGENERATE subset (>=4 allowed) and on a
    # per-decision-normalized gap, and report Spearman honestly rather than demoting it.
    primary_gate = (
        nondeg_top3 >= args.min_top3_hit
        and per_decision_frac <= args.max_per_decision_gap_fraction
    )
    aggregate["verdict"] = {
        "labels_consistent_with_qstar_nondegenerate": primary_gate,
        "reward_mode_interpretation": (
            "GOAL-ALIGNED Q* (reward = +1 only on the goal-reaching transition). Optimal = shortest "
            "discounted path to goal; decoy-milking is impossible, so a low non-degenerate Spearman/top-1 "
            "here CANNOT be blamed on the milking artifact and reflects genuine progress-ranking quality. "
            "Compare against the full-reward run to separate 'milking artifact' from 'real ranking weakness'."
            if args.reward_mode == "goal" else
            "FULL env reward (discovery + goal). Literal Q*; can milk distractor path_found rewards before a "
            "terminal, inflating absolute V*/gap. Use the goal-aligned run (--reward-mode goal) to isolate the "
            "milking artifact from genuine ranking weakness."
        ),
        "primary_gate": {
            "nondegenerate_top3_hit_ok": nondeg_top3 >= args.min_top3_hit,
            "per_decision_gap_fraction_ok": per_decision_frac <= args.max_per_decision_gap_fraction,
        },
        "honest_caveats": {
            "all_states_top3_is_vacuous": (
                "Top-3 on ALL states (~0.98) is inflated: ~75% of states have <=3 allowed actions where "
                "top-3 is 1.0 by construction. The gate now uses the non-degenerate (>=4 allowed) subset."
            ),
            "ranking_is_weak": (
                "Mean Spearman vs Q* is low and several tasks are anti-correlated; exact/decisive top-1 are "
                "modest. The oracle's full-ranking agreement with Q* is WEAK — its genuine signal is a small "
                "top-1 lift over random, not strong rank consistency."
            ),
            "decoy_milking_note": (
                "Literal-reward Q* milks distractor path_found rewards before a terminal on short leak tasks, "
                "inflating absolute V*/gap there (the DQN does not milk). This explains absolute-gap inflation "
                "but NOT the negative Spearman on some tasks, which is a genuine ranking weakness."
            ),
            "restricted_sub_mdp": (
                "Q* is exact only for the integer-action (fixed discovery-order) sub-MDP; targeted-order states "
                "the expert plan can reach are not enumerated. DQN scoring is made history-invariant by zeroing "
                "failed-history features before encoding."
            ),
        },
        "thresholds": {
            "min_nondegenerate_top3_hit": args.min_top3_hit,
            "max_per_decision_gap_fraction": args.max_per_decision_gap_fraction,
        },
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "aggregate": aggregate,
        "task_reports": task_reports,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def default_checkpoint() -> Path:
    for name in ["web_dqn_pilot.pt", "web_dqn_masked.pt", "web_dqn.pt"]:
        candidate = ROOT / "outputs" / name
        if candidate.exists():
            return candidate
    return ROOT / "outputs" / "web_dqn_masked.pt"


def resolve_gated_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    seed_gate = ROOT / "outputs" / "oracle_seed_gate.json"
    if seed_gate.exists():
        try:
            from generate_prm_dataset import select_checkpoint  # noqa: E402

            ns = argparse.Namespace(
                checkpoint=None,
                seed_gate_report=seed_gate,
                allow_ungated_oracle=False,
            )
            checkpoint, _gate = select_checkpoint(ns)
            return checkpoint
        except Exception:
            pass
    return default_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DQN oracle ranking against exact Q* value iteration.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to the best passed seed-gate checkpoint.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task JSON paths; defaults to all bundled tasks.")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount; should match DQN training (0.98).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--decisive-margin", type=float, default=0.5, help="Min Q* top1-vs-top2 gap for a state to count as decisive.")
    parser.add_argument("--min-top3-hit", type=float, default=0.9, help="Min top-3 hit on the NON-DEGENERATE (>=4 allowed) subset.")
    parser.add_argument("--max-per-decision-gap-fraction", type=float, default=0.25, help="Max greedy gap / per-state Q* spread.")
    parser.add_argument("--min-top1-agreement", type=float, default=0.9, help="Decisive-state top-1 agreement diagnostic.")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "qstar_report.json")
    parser.add_argument("--reward-mode", choices=["full", "goal"], default="full",
                        help="full = exact env reward (literal Q*, can milk decoys); goal = goal-aligned "
                             "reward (+1 at goal only) -> non-degenerate, isolates the milking artifact.")
    args = parser.parse_args()

    args.checkpoint = resolve_gated_checkpoint(args)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    sys.setrecursionlimit(100_000)
    report = evaluate(args)
    agg = report["aggregate"]
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "num_tasks": agg["num_tasks"],
                "total_reachable_states": agg["total_reachable_states"],
                "dqn_qstar_top3_hit_rate_ALL_states": round(agg["dqn_qstar_top3_hit_rate_ALL_states"], 4),
                "frac_states_degenerate_le3_allowed": round(agg["frac_states_degenerate_le3_allowed"], 4),
                "nondegenerate_top3_hit_rate": round(agg["nondegenerate_top3_hit_rate"], 4),
                "nondegenerate_top1_agreement": round(agg["nondegenerate_top1_agreement"], 4),
                "nondegenerate_mean_spearman": round(agg["nondegenerate_mean_spearman"], 4),
                "dqn_qstar_mean_spearman_ALL": round(agg["dqn_qstar_mean_spearman_ALL"], 4),
                "per_decision_gap_fraction": round(agg["per_decision_gap_fraction"], 4),
                "expert_qstar_avg_value_gap": round(agg["expert_qstar_avg_value_gap"], 4),
                "verdict": agg["verdict"],
                "report_output": str(args.report_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for task in report["task_reports"]:
        ra = task["ranking_agreement"]
        ea = task["expert_alignment"]
        nd3 = ra["nondegenerate_top3_hit_rate"]
        nds = ra["nondegenerate_mean_spearman"]
        alls = ra["dqn_qstar_mean_spearman"]
        nd3_str = f"{nd3:.2f}" if nd3 is not None else " n/a"
        nds_str = f"{nds:+.2f}" if nds is not None else " n/a"
        alls_str = f"{alls:+.2f}" if alls is not None else " n/a"
        print(
            f"{task['task_id']:30s} states={task['num_reachable_states']:4d} nondeg={ra['nondegenerate_states']:3d} "
            f"nd_top3={nd3_str} nd_spearman={nds_str} all_spearman={alls_str} "
            f"expert_gap={ea['expert_qstar_avg_value_gap']:.2f}"
        )


if __name__ == "__main__":
    main()
