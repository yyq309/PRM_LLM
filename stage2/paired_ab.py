"""PAIRED reranker A/B with common random numbers (CRN) — the variance-reduced version of the live A/B.

The unpaired live A/B (stage2.live_ab_trials / ablation_rerank) runs each arm as an INDEPENDENT
stochastic LLM rollout, so part of the arm-to-arm gap is proposer-sampling luck, not the ranking. Here,
for each (box, trial) we build ONE shared `CachingProposer` and run every rerank mode through it: the LLM
is called once per UNIQUE decision state and the result is REUSED across arms, so wherever the arms'
states coincide (always step 0 + the locked prefix) they rank the IDENTICAL candidate set and the only
difference is the rerank function. This cancels proposer-draw variance on the shared portion and ~halves
LLM calls.

Honest scope (see CachingProposer): exact pairing holds only on shared states; once two policies pick
differently their states (hence candidate sets) MUST differ and those steps query fresh — correct
sequential CRN, not total pairing. The report includes the cache hit-rate so the degree of pairing is
explicit.

    # offline self-check (no key, no formal experiment):
    python -m stage2.paired_ab --proposer target --executor dryrun --trials 2 --budget 8 \
        --boxes ThinkPHP-5-rce Struts2-S2-048 --modes llm_only prm random oracle

    # live formal run (later; needs DEEPSEEK_API_KEY + auth):
    python -m stage2.paired_ab --proposer llm --model deepseek-chat --trials 5 --budget 12 \
        --executor live --confirmed-isolated
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.engagement import (run_engagement, LLMProposer, TargetAwareProposer,  # noqa: E402
                               CachingProposer, _make_executor)
from stage2.eta import load_target  # noqa: E402
from stage2.safety import AuditLog  # noqa: E402
from stage2.stats_analysis import stratified_permutation, wilson  # noqa: E402

MODES = ["llm_only", "prm", "random", "oracle"]

# (descriptor file, label, full_goal_reachable) — default to the exploit-proposable boxes
TARGETS = [
    ("thinkphp-5-rce.json", "ThinkPHP-5-rce", True),
    ("thinkphp-5023-rce.json", "ThinkPHP-5.0.23", True),
    ("struts2-s2-048.json", "Struts2-S2-048", True),
    ("struts2-s2-045.json", "Struts2-S2-045", True),
    ("drupal-cve-2018-7600.json", "Drupalgeddon2", True),
    ("joomla-cve-2017-8917.json", "Joomla-8917-sqli", True),
    ("tomcat-cve-2017-12615.json", "Tomcat-12615", True),
    ("php-cgi-cve-2012-1823.json", "php-cgi-2012-1823", True),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired reranker A/B (shared-candidate CRN).")
    ap.add_argument("--proposer", choices=["target", "llm"], default="target")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--boxes", nargs="+", default=None)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--executor", choices=["dryrun", "live"], default="dryrun")
    ap.add_argument("--confirmed-isolated", action="store_true")
    ap.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "stage2_paired_ab.json")
    args = ap.parse_args()
    use_llm = args.proposer == "llm"
    modes = list(args.modes)
    targets = [t for t in TARGETS if (args.boxes is None or t[1] in args.boxes)]

    prm = joblib.load(ROOT / "outputs" / "prm_strong.joblib")
    audit = AuditLog(ROOT / "outputs" / "stage2_paired_audit.jsonl")

    episodes: list[dict] = []           # one per (box, trial, mode)
    per_box: dict = {}
    cache_hit_rates: list[float] = []

    for fn, label, reachable in targets:
        path = ROOT / "stage2" / "targets" / fn
        if not path.exists():
            print(f"  (missing descriptor {fn})", flush=True)
            continue
        desc = load_target(path)
        per_box[label] = {m: {"goal": 0, "prog": 0, "tot": 0} for m in modes}
        for trial in range(args.trials):
            # ONE shared cache per (box, trial): the LLM is sampled once per unique state and reused
            # across all modes -> shared candidate sets wherever the arms' states coincide.
            inner = (LLMProposer(model=args.model, temperature=args.temperature) if use_llm
                     else TargetAwareProposer(desc))
            shared = CachingProposer(inner)
            for mode in modes:
                ex = _make_executor(args.executor, desc, audit, None, args.confirmed_isolated)
                try:
                    r = run_engagement(desc, prm=prm, executor=ex, proposer=shared, mode=mode,
                                       budget=args.budget, audit=audit, permissive_guard=True,
                                       rerank_seed=trial, patience=args.patience)
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:20s} trial={trial} {mode:10s} ERR {type(e).__name__}: {str(e)[:50]}", flush=True)
                    continue
                b = per_box[label][mode]
                b["goal"] += int(bool(r["goal_reached"]))
                b["prog"] += r["progress_steps"]
                b["tot"] += r["steps_taken"]
                episodes.append({"box": label, "arm": mode, "reachable": reachable,
                                 "goal": int(bool(r["goal_reached"])), "_one": 1,
                                 "prog_steps": r["progress_steps"], "total_steps": r["steps_taken"]})
            cache_hit_rates.append(shared.cache_stats()["hit_rate"])
        done = per_box[label]
        print(f"{label:20s} " + "  ".join(
            f"{m}:{done[m]['prog']}/{done[m]['tot']}={done[m]['prog']/max(done[m]['tot'],1):.0%}" for m in modes),
            flush=True)

    full = [e for e in episodes if e["reachable"]]
    pooled = {}
    for m in modes:
        rows = [e for e in full if e["arm"] == m]
        pk = sum(e["prog_steps"] for e in rows); pn = sum(e["total_steps"] for e in rows)
        gk = sum(e["goal"] for e in rows); gn = len(rows)
        lo, hi = wilson(pk, pn)
        pooled[m] = {"progress": [pk, pn], "progress_rate": round(pk / max(pn, 1), 3),
                     "progress_ci95": [round(lo, 3), round(hi, 3)],
                     "goal": [gk, gn], "goal_rate": round(gk / max(gn, 1), 3)}

    tests = {}
    for m in modes:
        if m == "prm":
            continue
        pair = [{**e, "arm": ("prm" if e["arm"] == "prm" else "llm_only")}
                for e in full if e["arm"] in ("prm", m)]
        if pair:
            obs, p = stratified_permutation(pair, "prog_steps", "total_steps")
            tests[f"prm_vs_{m}"] = {"delta_pp": round(obs * 100, 1), "perm_p_clustered": round(p, 4),
                                   "significant": bool(p < 0.05)}

    mean_hit = round(sum(cache_hit_rates) / max(len(cache_hit_rates), 1), 3)
    out = {"meta": {"design": "paired (shared-candidate CRN)", "proposer": args.proposer,
                    "model": args.model if use_llm else None, "modes": modes, "trials": args.trials,
                    "budget": args.budget, "executor": args.executor,
                    "boxes": [t[1] for t in targets], "n_episodes": len(episodes),
                    "mean_cache_hit_rate": mean_hit,
                    "note": ("CRN: LLM sampled once per unique state, shared across arms; exact pairing on "
                             "shared states (always step 0 + locked prefix), fresh after divergence")},
           "per_box": per_box, "pooled_full_goal": pooled, "permutation_prm_vs_baseline": tests}
    args.report_output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"PAIRED per-step progress by mode (full-goal boxes); mean cache hit-rate={mean_hit:.0%} "
          f"(= LLM calls saved by sharing):")
    for m in modes:
        pk, pn = pooled[m]["progress"]
        print(f"  {m:10s} {pk:4d}/{pn:4d}={pooled[m]['progress_rate']:.1%} "
              f"CI[{pooled[m]['progress_ci95'][0]:.2f},{pooled[m]['progress_ci95'][1]:.2f}] "
              f"goal {pooled[m]['goal'][0]}/{pooled[m]['goal'][1]}={pooled[m]['goal_rate']:.0%}")
    for k, v in tests.items():
        print(f"  {k:18s} Δ={v['delta_pp']:+5.1f}pp perm-p={v['perm_p_clustered']} "
              f"{'SIGNIFICANT' if v['significant'] else 'NS'}")
    print(f"report -> {args.report_output.name}")


if __name__ == "__main__":
    main()
