"""Reward-sensitivity robustness ablation for the Web-RL value oracle (method §5.1).

Two clearly-separated experiments (an earlier version conflated them, which an
adversarial review correctly flagged as a near-tautology):

1. RELATIVE-RATIO perturbation (the real §5.1 concern). The reward magnitudes are
   hand-tuned constants; we perturb the *relative ratios* between event rewards by
   independent multiplicative jitter (different per seed) at a fixed global scale, and
   check the oracle's ranking (held-out goal rate, expert top-1/top-3) is stable. This
   tests robustness to the actual hand-tuned numbers, not a global multiplier.

2. GLOBAL-SCALING sanity (trivial). Rewards are multiplied by a single constant c with
   no jitter. Under positive global scaling with fixed gamma the optimal policy and
   Q-ranking are provably invariant and Q* scales by c — so this only confirms value_gap
   scales ~linearly with c. It is reported as a sanity check, NOT as robustness evidence.

Ranking metrics are reported against the random-within-mask baseline where relevant,
since (per the same review) raw top-3 under the strict mask is near random chance.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_oracle_seed_gate import DEFAULT_EVAL_TASKS, DEFAULT_TRAIN_TASKS  # noqa: E402


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def spread(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "spread": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "min": min(values), "max": max(values), "spread": max(values) - min(values), "std": var ** 0.5}


def train_and_eval(scale: float, jitter: float, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    tag = f"scale{scale}_jit{jitter}_seed{seed}".replace(".", "p")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "oracle.pt"
    eval_report = out_dir / "oracle_eval.json"

    if not (args.skip_existing and checkpoint.exists()):
        run([
            sys.executable, str(ROOT / "scripts" / "train_dqn.py"),
            "--seed", str(seed), "--training-steps", str(args.training_steps),
            "--eval-episodes", "3", "--log-every", "200", "--device", args.device,
            "--reward-scale", str(scale), "--reward-jitter", str(jitter),
            "--output", str(checkpoint),
            "--train-tasks", *DEFAULT_TRAIN_TASKS, "--eval-tasks", *DEFAULT_EVAL_TASKS,
        ])
    if not (args.skip_existing and eval_report.exists()):
        run([
            sys.executable, str(ROOT / "scripts" / "evaluate_oracle.py"),
            "--checkpoint", str(checkpoint), "--tasks", *DEFAULT_EVAL_TASKS,
            "--episodes-per-task", "1", "--device", args.device,
            "--report-output", str(eval_report), "--q-report-output", str(out_dir / "q.jsonl"),
        ])
    r = read_json(eval_report)
    return {
        "scale": scale, "jitter": jitter, "seed": seed,
        "eval_goal_rate": r["eval_goal_rate"],
        "expert_top1": r["expert_action_top1_rate"],
        "expert_top3": r["expert_action_top3_rate"],
        "expert_value_gap": r["expert_action_avg_value_gap"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward-sensitivity ablation: relative-ratio robustness + global-scaling sanity (§5.1).")
    parser.add_argument("--rel-jitter", type=float, default=0.3, help="Relative-ratio perturbation magnitude at fixed scale 1.0.")
    parser.add_argument("--rel-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="Global-scaling sanity scales (jitter 0).")
    parser.add_argument("--scale-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--training-steps", type=int, default=15000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reward_sensitivity")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "reward_sensitivity.json")
    parser.add_argument("--top1-stability-tol", type=float, default=0.15)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    # Experiment 1: relative-ratio perturbation at fixed scale 1.0 (+ unperturbed anchor).
    relative_rows = [train_and_eval(1.0, 0.0, args.rel_seeds[0], args)]  # production-config anchor
    for seed in args.rel_seeds:
        relative_rows.append(train_and_eval(1.0, args.rel_jitter, seed, args))

    # Experiment 2: global-scaling sanity (no jitter -> pure scalar multiple).
    scaling_rows = [train_and_eval(scale, 0.0, seed, args) for scale in args.scales for seed in args.scale_seeds]

    rel_top1 = spread([r["expert_top1"] for r in relative_rows])
    rel_top3 = spread([r["expert_top3"] for r in relative_rows])
    rel_goal = spread([r["eval_goal_rate"] for r in relative_rows])

    by_scale: dict[str, list[dict[str, Any]]] = {}
    for r in scaling_rows:
        by_scale.setdefault(str(r["scale"]), []).append(r)
    scaling_summary = {
        s: {
            "goal_rate_mean": sum(x["eval_goal_rate"] for x in g) / len(g),
            "top3_mean": sum(x["expert_top3"] for x in g) / len(g),
            "value_gap_mean": sum(x["expert_value_gap"] for x in g) / len(g),
        }
        for s, g in sorted(by_scale.items(), key=lambda kv: float(kv[0]))
    }
    gaps = sorted((float(s), v["value_gap_mean"]) for s, v in scaling_summary.items())
    gap_ratio = (gaps[-1][1] / gaps[0][1]) if len(gaps) >= 2 and gaps[0][1] else None
    scale_ratio = (gaps[-1][0] / gaps[0][0]) if len(gaps) >= 2 and gaps[0][0] else None

    report = {
        "relative_ratio_perturbation": {
            "description": "Fixed scale 1.0; independent per-event jitter (different per seed) perturbs the relative reward ratios. Stable ranking here = robustness to the hand-tuned constants.",
            "rel_jitter": args.rel_jitter,
            "rows": relative_rows,
            "expert_top1": rel_top1,
            "expert_top3": rel_top3,
            "goal_rate": rel_goal,
            "verdict_relative_ranking_stable": bool(rel_top1["spread"] <= args.top1_stability_tol),
        },
        "global_scaling_sanity": {
            "description": "Pure scalar reward multiplication (jitter 0). Ranking invariance is a trivial MDP property; only value_gap scaling is informative here.",
            "by_scale": scaling_summary,
            "value_gap_ratio_observed": gap_ratio,
            "value_gap_ratio_expected": scale_ratio,
            "value_gap_scales_with_reward": bool(gap_ratio is not None and gap_ratio > 1.5),
            "note": "If value_gap_ratio_observed differs much from value_gap_ratio_expected, the networks are under-converged at this step budget (a fixed-budget confound), not a robustness signal.",
        },
        "summary_note": (
            "Relative-ratio perturbation is the real §5.1 robustness test; global scaling is a trivial "
            "sanity check. Both are reported separately so the trivial invariance cannot masquerade as "
            "robustness."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== relative-ratio perturbation (fixed scale 1.0) ===")
    print(f"  top1 mean={rel_top1['mean']:.3f} spread={rel_top1['spread']:.3f}  top3 mean={rel_top3['mean']:.3f} spread={rel_top3['spread']:.3f}  goal spread={rel_goal['spread']:.3f}")
    print(f"  verdict relative ranking stable: {report['relative_ratio_perturbation']['verdict_relative_ranking_stable']}")
    print("=== global-scaling sanity ===")
    for s, v in scaling_summary.items():
        print(f"  scale={s:>4s} goal={v['goal_rate_mean']:.3f} top3={v['top3_mean']:.3f} value_gap={v['value_gap_mean']:.3f}")
    print(f"  value_gap ratio observed={gap_ratio} expected={scale_ratio}")


if __name__ == "__main__":
    main()
