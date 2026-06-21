"""One-command reproducer for the WebAttackSim TRAINING STAGE (stage 1).

Runs the full label-transfer + evaluation pipeline in dependency order and prints a PASS/FAIL
manifest. The expensive oracle retrain and the slowest evaluations are gated behind --include-slow
(they reproduce identical numbers deterministically and are already materialized in outputs/).

Order:  smoke -> coverage -> [oracle gate] -> PRM dataset -> baseline/strong/robust PRM
        -> [multiseed + joint] -> honest_eval -> [qstar] -> learning curve -> [newfam samples]
        -> zero-shot -> error-action -> [MC return] -> MC blend -> trajectory credit
        -> [closed-loop policy] -> summary -> tests

This is stage-1 only: NO adapter / real-target / inference. Use the canonical (pinned) oracle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the WebAttackSim training stage end-to-end.")
    parser.add_argument("--include-slow", action="store_true",
                        help="Also run the oracle retrain (80k x3), per-seed MC, verify_qstar, permissive policy, multiseed/joint.")
    parser.add_argument("--retrain-oracle", action="store_true", help="Force the 80k x3 permissive oracle gate (very slow).")
    parser.add_argument("--stop-on-fail", action="store_true", help="Abort at the first failing step.")
    args = parser.parse_args()

    NEW = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tasks").glob("gen_25*_*.json"))
    NEW += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tasks").glob("gen_26*_*.json"))

    # (name, argv, slow)
    plan: list[tuple[str, list[str], bool]] = [
        ("smoke_test_env", [PY, "scripts/smoke_test_env.py"], False),
        ("coverage_audit", [PY, "scripts/coverage_audit.py"], False),
        ("oracle_seed_gate(80k)", [PY, "scripts/run_oracle_seed_gate.py", "--seeds", "0", "1", "2",
                                   "--training-steps", "80000", "--train-no-action-mask",
                                   "--output-dir", "outputs/seed_gate_12fam_80k",
                                   "--aggregate-output", "outputs/oracle_seed_gate_12fam_80k.json",
                                   # hard mode: tight budget makes even masked-greedy miss the goal
                                   # sometimes, so goal_rate ~0.6 is EXPECTED and more meaningful than
                                   # the old saturated 1.0. Gate on top-3 + gap, not a 1.0 goal_rate.
                                   "--min-goal-rate", "0.6", "--min-expert-top3-rate", "0.6",
                                   "--max-expert-avg-gap", "6.0"], True),
        ("generate_prm_dataset", [PY, "scripts/generate_prm_dataset.py"], False),
        ("train_prm_baseline", [PY, "scripts/train_prm_baseline.py"], False),
        ("train_prm_strong(+calib)", [PY, "scripts/train_prm_strong.py"], False),
        ("train_prm_robust(+OOD)", [PY, "scripts/train_prm_robust.py"], False),
        ("honest_eval", [PY, "scripts/honest_eval.py"], False),
        ("family_learning_curve", [PY, "scripts/family_learning_curve.py"], False),
        ("error_action_eval", [PY, "scripts/error_action_eval.py"], False),
        ("mc_return_labels", [PY, "scripts/mc_return_labels.py"], True),
        ("mc_blend_train", [PY, "scripts/mc_blend_train.py"], False),
        ("trajectory_credit_eval", [PY, "scripts/trajectory_credit_eval.py"], False),
        ("verify_qstar(full)", [PY, "scripts/verify_qstar.py"], True),
        ("verify_qstar(goal)", [PY, "scripts/verify_qstar.py", "--reward-mode", "goal",
                                "--report-output", "outputs/qstar_report_goal.json"], True),
        ("newfam_samples", [PY, "scripts/generate_prm_dataset.py", "--train-tasks", "tasks/gen_201_leak_file.json",
                            "--heldout-tasks", *NEW, "--train-output", "outputs/_tmp_train_nf.jsonl",
                            "--heldout-output", "outputs/prm_samples_newfam.jsonl",
                            "--ranking-output", "outputs/_tmp_rank_nf.json", "--summary-output", "outputs/_tmp_sum_nf.json"], True),
        ("eval_new_family_zeroshot", [PY, "scripts/eval_new_family_zeroshot.py"], False),
        ("evaluate_prm_policy", [PY, "scripts/evaluate_prm_policy.py"], False),
        ("evaluate_prm_policy(strong)", [PY, "scripts/evaluate_prm_policy.py", "--prm-model", "outputs/prm_strong.joblib",
                                         "--policies", "prm", "oracle", "random_valid",
                                         "--report-output", "outputs/prm_policy_eval_strong.json"], True),
        ("build_training_summary", [PY, "scripts/build_training_summary.py"], False),
        ("pytest(not slow)", [PY, "-m", "pytest", "tests/", "-m", "not slow", "-q"], False),
    ]

    results = []
    for name, argv, slow in plan:
        is_oracle = name.startswith("oracle_seed_gate")
        # The oracle retrain is gated SEPARATELY by --retrain-oracle (so --include-slow can run all the
        # slow EVALS without paying for an 80kx3 retrain when the canonical oracle is already trained).
        run_this = (not slow) or (args.retrain_oracle if is_oracle else args.include_slow)
        if not run_this:
            results.append((name, "SKIP", 0.0))
            print(f"[SKIP] {name}")
            continue
        print(f"[RUN ] {name}: {' '.join(argv)}", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run(argv, cwd=ROOT)
        dt = time.monotonic() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append((name, status, dt))
        print(f"[{status}] {name}  ({dt:.1f}s)", flush=True)
        if status == "FAIL" and args.stop_on_fail:
            break

    # cleanup temp newfam artifacts
    for tmp in ["outputs/_tmp_train_nf.jsonl", "outputs/_tmp_rank_nf.json", "outputs/_tmp_sum_nf.json"]:
        p = ROOT / tmp
        if p.exists():
            p.unlink()

    print("\n==== TRAINING-STAGE MANIFEST ====")
    npass = sum(1 for _, s, _ in results if s == "PASS")
    nfail = sum(1 for _, s, _ in results if s == "FAIL")
    nskip = sum(1 for _, s, _ in results if s == "SKIP")
    for name, status, dt in results:
        suffix = f"  ({dt:.1f}s)" if status != "SKIP" else "  (gated by --include-slow)"
        print(f"  {status:4s}  {name}{suffix}")
    print(f"  ---- {npass} pass / {nfail} fail / {nskip} skipped ----")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
