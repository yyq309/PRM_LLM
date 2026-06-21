"""Rerank-ISOLATION ablation: is the Stage-2 gain actually the PRM, or the harness around it?

The live A/B varies the LLM proposer (stochastic), so part of any prm-vs-llm_only gap could be
proposer luck, the no-progress guard, the permissive guard, the target recipe, or candidate order --
not the PRM's ranking. This ablation removes every one of those confounds:

  * the PROPOSER is held FIXED and deterministic (TargetAwareProposer -- emits the box's declared
    candidate surface), so proposer quality/luck is constant across arms;
  * the same per-trial candidate-order SHUFFLE seed is replayed for every rerank mode, so all modes
    rank the IDENTICAL shuffled candidate list (a paired design) -- candidate order is constant;
  * the guards / recipe / executor / φ are identical across arms;
  * the ONLY thing that varies between arms is the rerank FUNCTION:
        random        -- rank carries no information (floor)
        shuffled_prm  -- the PRM's score DISTRIBUTION but the score->action mapping destroyed
                         (controls for "PRM just always prefers one action type")
        heuristic     -- a fixed hand priority over action types
        prm           -- the abstract-trained PRM  (the method)
        oracle        -- goal-aware next-milestone ranking (privileged upper bound / headroom)

If prm ~ random, the gain is NOT the PRM. If prm > random/shuffled_prm/heuristic and approaches
oracle, the gain IS the PRM's learned ranking. Runs LIVE against the standing containers (no DeepSeek
key needed -- the proposer is offline). Per-step progress is tested with the same episode-clustered
permutation test as stage2.stats_analysis.

    STAGE2_LIVE_AUTHORIZED=i-own-this-isolated-authorized-lab \
    python -m stage2.ablation_rerank --seeds 8 --budget 12 --confirmed-isolated
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.engagement import (run_engagement, TargetAwareProposer, LLMProposer,  # noqa: E402
                               _make_executor)
from stage2.eta import load_target  # noqa: E402
from stage2.safety import AuditLog  # noqa: E402
from stage2.stats_analysis import stratified_permutation, wilson  # noqa: E402

MODES = ["random", "shuffled_prm", "heuristic", "prm", "oracle"]

# (descriptor file, label, full_goal_reachable)  -- same set as stats_analysis, auth-milestone flagged
TARGETS = [
    ("thinkphp-5-rce.json",      "ThinkPHP-5-rce",      True),
    ("thinkphp-5023-rce.json",   "ThinkPHP-5.0.23",     True),
    ("struts2-s2-048.json",      "Struts2-S2-048",      True),
    ("struts2-s2-045.json",      "Struts2-S2-045",      True),
    ("drupal-cve-2018-7600.json","Drupalgeddon2",       True),
    ("tomcat-cve-2017-12615.json","Tomcat-12615",       True),
    ("joomla-cve-2017-8917.json","Joomla-8917-sqli",    True),
    ("php-cgi-cve-2012-1823.json","php-cgi-2012-1823",   True),
    ("php-inclusion-lfi.json",   "php-inclusion-LFI",   True),
    ("rails-cve-2019-5418.json", "Rails-5418-fileread", True),
    ("weblogic-weak-password.json","WebLogic-weakpw",   False),
    ("gitea-1.4-rce.json",       "Gitea-1.4",           False),
]


def main() -> None:
    global MODES
    ap = argparse.ArgumentParser(description="Rerank-isolation ablation (live).")
    ap.add_argument("--seeds", type=int, default=8, help="trials per (box,mode); rerank-rng/shuffle seed")
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--executor", choices=["dryrun", "live"], default="live")
    ap.add_argument("--proposer", choices=["target", "llm"], default="target",
                    help="target=deterministic stand-in (key-free, paired shuffle); llm=real DeepSeek "
                         "proposer (needs DEEPSEEK_API_KEY; its stochasticity supplies trial variation)")
    ap.add_argument("--model", default="deepseek-chat", help="DeepSeek model for --proposer llm")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", default=MODES, help="subset of rerank modes to run")
    ap.add_argument("--boxes", nargs="+", default=None,
                    help="subset of box labels (default: all 12); the LLM run focuses on exploit-proposable boxes")
    ap.add_argument("--confirmed-isolated", action="store_true")
    ap.add_argument("--targets-dir", type=Path, default=ROOT / "stage2" / "targets")
    ap.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "stage2_ablation_rerank.json")
    args = ap.parse_args()

    MODES = list(args.modes)
    use_llm = args.proposer == "llm"
    targets_run = [t for t in TARGETS if (args.boxes is None or t[1] in args.boxes)]

    prm = joblib.load(ROOT / "outputs" / "prm_strong.joblib")
    audit = AuditLog(ROOT / "outputs" / "stage2_ablation_audit.jsonl")
    # episode rows: one per (box, mode, seed) engagement -> for the clustered permutation test
    episodes: list[dict] = []
    per_box: dict = {}

    for fn, label, reachable in targets_run:
        path = args.targets_dir / fn
        if not path.exists():
            print(f"  (missing descriptor {fn})", flush=True)
            continue
        desc = load_target(path)
        per_box[label] = {m: {"goal": 0, "prog": 0, "tot": 0, "milestone": []} for m in MODES}
        for seed in range(args.seeds):
            for mode in MODES:
                ex = _make_executor(args.executor, desc, audit, None, args.confirmed_isolated)
                prop = (LLMProposer(model=args.model, temperature=args.temperature) if use_llm
                        else TargetAwareProposer(desc))
                try:
                    r = run_engagement(desc, prm=prm, executor=ex, proposer=prop,
                                       mode=mode, budget=args.budget, audit=audit, permissive_guard=True,
                                       rerank_seed=seed, shuffle_candidates=not use_llm)
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:20s} seed={seed} {mode:13s} ERR {type(e).__name__}: {str(e)[:50]}", flush=True)
                    continue
                ms = (5 if r["reached_root"] else 4 if r["read_any_file"] else 3 if r["reached_command_exec"]
                      else 2 if r["reached_shell"] else 0)
                b = per_box[label][mode]
                b["goal"] += int(bool(r["goal_reached"])); b["prog"] += r["progress_steps"]
                b["tot"] += r["steps_taken"]; b["milestone"].append(ms)
                episodes.append({"box": label, "arm": mode, "reachable": reachable,
                                 "goal": int(bool(r["goal_reached"])), "_one": 1,
                                 "prog_steps": r["progress_steps"], "total_steps": r["steps_taken"]})
        done = per_box[label]
        print(f"{label:20s} " + "  ".join(
            f"{m}:{done[m]['prog']}/{done[m]['tot']}={done[m]['prog']/max(done[m]['tot'],1):.0%}" for m in MODES),
            flush=True)

    # ---- pooled per-step progress per mode (full-goal boxes) + permutation prm-vs-baseline ----
    full = [e for e in episodes if e["reachable"]]
    pooled = {}
    for m in MODES:
        rows = [e for e in full if e["arm"] == m]
        pk = sum(e["prog_steps"] for e in rows); pn = sum(e["total_steps"] for e in rows)
        gk = sum(e["goal"] for e in rows); gn = len(rows)
        lo, hi = wilson(pk, pn)
        pooled[m] = {"progress": [pk, pn], "progress_rate": round(pk / max(pn, 1), 3),
                     "progress_ci95": [round(lo, 3), round(hi, 3)],
                     "goal": [gk, gn], "goal_rate": round(gk / max(gn, 1), 3)}

    # permutation: prm vs each other mode, episode-clustered (box as stratum), per-step progress
    tests = {}
    for m in MODES:
        if m == "prm":
            continue
        pair = [e for e in full if e["arm"] in ("prm", m)]
        for e in pair:
            e["arm2"] = "prm" if e["arm"] == "prm" else "llm_only"  # reuse stratified_permutation's arm keys
        rows2 = [{**e, "arm": e["arm2"]} for e in pair]
        obs, p = stratified_permutation(rows2, "prog_steps", "total_steps")
        tests[f"prm_vs_{m}"] = {"delta_pp": round(obs * 100, 1), "perm_p_clustered": round(p, 4),
                                "significant": bool(p < 0.05)}

    # per-episode permutation prm-vs-baseline (goal-reach), episode-clustered by box, for the high-N runs
    goal_tests = {}
    for m in MODES:
        if m == "prm":
            continue
        pair = [{**e, "arm": ("prm" if e["arm"] == "prm" else "llm_only")}
                for e in full if e["arm"] in ("prm", m)]
        for e in pair:
            e["_one"] = 1
        if pair:
            obs, p = stratified_permutation(pair, "goal", "_one")
            goal_tests[f"prm_vs_{m}"] = {"delta_pp": round(obs * 100, 1),
                                         "perm_p_clustered": round(p, 4), "significant": bool(p < 0.05)}

    out = {"meta": {"proposer": (f"LLMProposer({args.model},temp={args.temperature})" if use_llm
                                 else "TargetAwareProposer(deterministic)"),
                    "executor": args.executor, "seeds": args.seeds, "budget": args.budget, "modes": MODES,
                    "boxes": [t[1] for t in targets_run], "n_episodes": len(episodes),
                    "note": ("reranker isolation; " + ("real LLM proposer (stochastic trial variation)"
                             if use_llm else "key-free deterministic proposer, paired shuffle seeds"))},
           "per_box": per_box, "pooled_full_goal": pooled,
           "permutation_prm_vs_baseline": tests, "goal_permutation_prm_vs_baseline": goal_tests,
           "episodes": [{"box": e["box"], "arm": e["arm"], "reachable": e["reachable"],
                         "goal": e["goal"], "prog_steps": e["prog_steps"],
                         "total_steps": e["total_steps"]} for e in episodes]}
    args.report_output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("POOLED per-step progress by rerank mode (full-goal boxes), prm vs each baseline:")
    for m in MODES:
        pk, pn = pooled[m]["progress"]
        print(f"  {m:13s} {pk:4d}/{pn:4d} = {pooled[m]['progress_rate']:.1%}  "
              f"CI[{pooled[m]['progress_ci95'][0]:.2f},{pooled[m]['progress_ci95'][1]:.2f}]  "
              f"goal {pooled[m]['goal'][0]}/{pooled[m]['goal'][1]}={pooled[m]['goal_rate']:.0%}")
    print("\n  prm vs baseline (episode-clustered permutation, per-step progress):")
    for k, v in tests.items():
        print(f"    {k:22s} Δ={v['delta_pp']:+5.1f}pp  perm-p={v['perm_p_clustered']}  "
              f"{'SIGNIFICANT' if v['significant'] else 'NS'}")
    try:
        shown = args.report_output.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.report_output
    print(f"\nreport -> {shown}")


if __name__ == "__main__":
    main()
