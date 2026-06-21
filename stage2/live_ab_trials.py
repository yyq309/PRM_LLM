"""Multi-trial live A/B: PRM-rerank vs proposer-order, over N independent trials, with goal-reach
rates + Wilson CIs. A single A/B run is a stochastic anecdote (the LLM proposer is non-deterministic);
this aggregates N trials so the comparison is credible, not luck.

    DEEPSEEK_API_KEY=... STAGE2_LIVE_AUTHORIZED=... \
    python -m stage2.live_ab_trials --target stage2/targets/thinkphp-5-rce.json \
        --proposer llm --model deepseek-v4-pro --trials 6 --budget 14 --confirmed-isolated
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib  # noqa: E402

from stage2.engagement import (run_engagement, LLMProposer, TargetAwareProposer, StateProposer,  # noqa: E402
                               CachingProposer, _make_executor)
from stage2.eta import load_target  # noqa: E402
from stage2.safety import AuditLog  # noqa: E402
from stage2 import vm_reset  # noqa: E402


def _revert_vm(desc: dict) -> dict:
    """Full-chain VM boxes that MUTATE state (Raven UDF, Symfonos PATH-hijack) must be reverted to the
    'clean' snapshot before each engagement so one arm cannot pollute the next. DC-1/Toppo are
    non-mutating so --reset-vm is optional for them."""
    reg = vm_reset._load()
    vm = desc.get("vm", {})
    t = {"label": desc.get("name", "vm"), "vmx": vm.get("vmx"), "snapshot": vm.get("snapshot", "clean"),
         "ip": vm.get("ip"), "port": desc.get("port", 80), "healthcheck": "/",
         "expect_status": desc.get("expect_status", [200, 301, 302, 403])}
    return vm_reset.reset_one(t, vm_reset._vmrun_path(reg))


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - half) / d), min(1.0, (c + half) / d))


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-trial live A/B with goal-reach rates + CIs.")
    p.add_argument("--target", type=Path, default=ROOT / "stage2" / "targets" / "thinkphp-5-rce.json")
    p.add_argument("--prm-model", type=Path, default=ROOT / "outputs" / "prm_strong.joblib")
    p.add_argument("--executor", choices=["dryrun", "live"], default="live")
    p.add_argument("--proposer", choices=["state", "target", "llm"], default="llm")
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--budget", type=int, default=14)
    p.add_argument("--confirmed-isolated", action="store_true")
    p.add_argument("--patience", type=int, default=4,
                   help="global no-progress circuit-breaker (raise for genuine multi-step chains; 0 disables)")
    p.add_argument("--seed", type=int, default=0, help="seed for the per-trial arm-order shuffle (reproducible)")
    p.add_argument("--temperature", type=float, default=0.5, help="LLM proposer temperature (recorded)")
    p.add_argument("--fixed-arm-order", action="store_true",
                   help="keep prm-then-llm_only every trial (default randomizes to remove order bias)")
    p.add_argument("--unpaired", action="store_true",
                   help="disable the per-trial shared CachingProposer (CRN). Default is PAIRED: arms share "
                        "the candidate set on coincident states -> ~half the LLM calls + lower variance.")
    p.add_argument("--reset-vm", action="store_true",
                   help="revert the full_vm 'clean' snapshot before each engagement (needed for state-mutating "
                        "boxes: Raven UDF / Symfonos PATH-hijack; optional for non-mutating DC-1/Toppo).")
    p.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "stage2_live_ab_trials.json")
    args = p.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    desc = load_target(args.target)
    prm = joblib.load(args.prm_model)
    audit = AuditLog(ROOT / "outputs" / "stage2_ab_trials_audit.jsonl")

    def proposer():
        if args.proposer == "llm":
            return LLMProposer(model=args.model, temperature=args.temperature)
        if args.proposer == "target":
            return TargetAwareProposer(desc)
        return StateProposer()

    arms = {"prm": [], "llm_only": []}
    errors = 0
    arm_orders = []  # record the (possibly randomized) per-trial arm order to expose any order bias
    cache_hits = []  # per-trial CachingProposer hit-rate (the LLM-call savings from CRN pairing)
    rng = random.Random(args.seed)
    for i in range(args.trials):
        order = ["prm", "llm_only"]
        if not args.fixed_arm_order:
            rng.shuffle(order)            # randomize arm order each trial -> removes first-mover bias
        arm_orders.append(order[:])
        # PAIRED (default): one shared CachingProposer per trial -> the two arms reuse the SAME candidate
        # set on coincident states (CRN), ~halving LLM calls and cancelling proposer-draw variance.
        trial_prop = None if args.unpaired else CachingProposer(proposer())
        for mode in order:
            if args.reset_vm and desc.get("kind") == "full_vm":
                rv = _revert_vm(desc)
                print(f"  [reset-vm] {rv['label']} ok={rv['ok']} status={rv.get('status')}", flush=True)
            ex = _make_executor(args.executor, desc, audit, None, args.confirmed_isolated)
            # a transient network drop (DeepSeek SSL EOF) must not kill the whole study — record the
            # trial as errored and continue, so a few blips degrade N rather than abort everything.
            try:
                r = run_engagement(desc, prm=prm, executor=ex, proposer=(trial_prop or proposer()), mode=mode,
                                   budget=args.budget, audit=audit, permissive_guard=True,
                                   patience=args.patience)
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"trial {i+1}/{args.trials} [{mode:8s}] ERRORED ({type(e).__name__}: {str(e)[:60]})", flush=True)
                continue
            r.pop("final_observation", None)  # keep `steps` for per-step pooling; drop the bulky obs
            arms[mode].append(r)
            print(f"trial {i+1}/{args.trials} [{mode:8s}] goal={r['goal_reached']} "
                  f"steps={r['steps_taken']} progress_rate={r['per_step_progress_rate']:.2f} "
                  f"shell={r['reached_shell']} cmd={r['reached_command_exec']} read={bool(r['read_any_file'])}",
                  flush=True)
        if trial_prop is not None:
            cache_hits.append(trial_prop.cache_stats()["hit_rate"])

    def _mean(rows, k):
        return round(sum(x.get(k, 0) for x in rows) / max(len(rows), 1), 3)

    def summ(rows):
        n = len(rows)
        g = sum(1 for x in rows if x["goal_reached"])
        lo, hi = _wilson(g, n)
        gsteps = [x["steps_taken"] for x in rows if x["goal_reached"]]
        # per-step progress rate pooled over ALL steps (high-N; far more sensitive than per-episode goal)
        prog = sum(x["progress_steps"] for x in rows)
        tot = sum(x["steps_taken"] for x in rows)
        plo, phi = _wilson(prog, tot)
        # C-B phase-split: per-step progress in the WEB/recon phase (milestone_before<2, no foothold yet)
        # vs the LOCAL phase (milestone_before>=2, post-foothold). Shows WHERE the reranker's value lives.
        wp = wt = lp = lt = 0
        for x in rows:
            for s in x.get("steps", []):
                if s.get("milestone_before", 0) >= 2:
                    lt += 1; lp += int(bool(s.get("made_progress")))
                else:
                    wt += 1; wp += int(bool(s.get("made_progress")))
        return {"n": n,
                # graded milestones (partial credit)
                "goal_reached": g, "goal_rate": round(g / max(n, 1), 3),
                "goal_rate_ci95": [round(lo, 3), round(hi, 3)],
                "shell_rate": round(sum(1 for x in rows if x["reached_shell"]) / max(n, 1), 3),
                "command_exec_rate": round(sum(1 for x in rows if x["reached_command_exec"]) / max(n, 1), 3),
                "file_read_rate": round(sum(1 for x in rows if x["read_any_file"]) / max(n, 1), 3),
                "root_rate": round(sum(1 for x in rows if x["reached_root"]) / max(n, 1), 3),
                # process / value-of-information (high-N per-step metric)
                "per_step_progress": {"progress_steps": prog, "total_steps": tot,
                                      "rate": round(prog / max(tot, 1), 3), "ci95": [round(plo, 3), round(phi, 3)]},
                "phase_split": {"web": {"progress": wp, "total": wt, "rate": round(wp / max(wt, 1), 3)},
                                "local": {"progress": lp, "total": lt, "rate": round(lp / max(lt, 1), 3)}},
                "mean_fields_gained": _mean(rows, "fields_gained_total"),
                "mean_distinct_productive_actions": _mean(rows, "distinct_productive_actions"),
                "mean_wasted_rate": _mean(rows, "wasted_rate"),
                "mean_steps_when_goal": round(sum(gsteps) / len(gsteps), 2) if gsteps else None,
                # cost
                "mean_proposer_calls": _mean(rows, "proposer_calls"),
                "mean_eta_executions": _mean(rows, "eta_executions"),
                # abstraction + safety
                "mean_live_out_of_abstraction_rate": _mean(rows, "live_out_of_abstraction_rate"),
                "total_gate_refusals": sum(x.get("gate_refusals", 0) for x in rows)}

    out = {"stage": "stage2_live_ab_trials", "target": desc["name"], "proposer": args.proposer,
           "model": args.model if args.proposer == "llm" else None, "trials": args.trials,
           "errored_trials": errors,
           # full reproducibility metadata (model/temp/seed/budget/target/timing/code version)
           "run_metadata": {
               "target_name": desc["name"], "target_url": desc.get("target"),
               "descriptor": str(args.target.name if hasattr(args.target, "name") else args.target),
               "proposer": args.proposer, "model": args.model if args.proposer == "llm" else None,
               "temperature": args.temperature if args.proposer == "llm" else None,
               "executor": args.executor, "budget": args.budget, "trials": args.trials,
               "arm_order_seed": args.seed, "arm_orders": arm_orders,
               "fixed_arm_order": bool(args.fixed_arm_order),
               "paired_crn": not args.unpaired,
               "mean_cache_hit_rate": round(sum(cache_hits) / len(cache_hits), 3) if cache_hits else None,
               "prm_model": str(args.prm_model.name), "started": started,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "psi_for_action_mapping": "enhanced", "psi_for_prm_features": "stage1_training_time",
               "goal_def": "command_execution AND a sensitive-file read (or root/flag file)",
               "deepseek_key_source": "DEEPSEEK_API_KEY env" if os.environ.get("DEEPSEEK_API_KEY") else "n/a"},
           "budget": args.budget, "prm": summ(arms["prm"]), "llm_only": summ(arms["llm_only"]),
           "per_trial": arms,
           "note": ("Live multi-trial A/B on ONE box. Goal = command execution + a sensitive-file read. "
                    "Small N: read the CIs, not the point estimates. A full uplift study spans several "
                    "boxes. The LLM proposer is stochastic (temp 0.5), which is the intended variation.")}
    args.report_output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== AGGREGATE (beyond success rate) ===")
    for arm in ("prm", "llm_only"):
        s = out[arm]
        ps = s["per_step_progress"]
        print(f"  {arm:8s} goal={s['goal_rate']:.0%} CI{s['goal_rate_ci95']} | "
              f"per-step-progress={ps['rate']:.0%} ({ps['progress_steps']}/{ps['total_steps']}) CI{ps['ci95']} | "
              f"milestones shell/cmd/file/root={s['shell_rate']:.0%}/{s['command_exec_rate']:.0%}/"
              f"{s['file_read_rate']:.0%}/{s['root_rate']:.0%} | "
              f"fields_gained={s['mean_fields_gained']} wasted={s['mean_wasted_rate']:.0%} | "
              f"cost calls/exec={s['mean_proposer_calls']}/{s['mean_eta_executions']} | "
              f"ooa_live={s['mean_live_out_of_abstraction_rate']:.0%} refusals={s['total_gate_refusals']}")
    print(f"report -> {args.report_output}")


if __name__ == "__main__":
    main()
