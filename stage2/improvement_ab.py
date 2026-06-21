"""A/B two engagement CONFIGS with the real LLM proposer, to test the two improvements the
LLM-memory investigation recommended:

  --compare memory    : baseline (poor memory: bool-only 3-step window + exhausted-type set)
                        vs rich_memory (ordered trace with evidence + per-type no-progress counts).
                        Question: does richer per-box memory reduce spinning (consecutive repeats /
                        wasted steps)?  [tests the user's hypothesis]
  --compare proposer  : baseline SYS prompt vs SYS_ENHANCED (fingerprint->named-CVE->exploit, curb
                        over-recon) — leak-free, leans on the LLM's own CVE knowledge.
                        Question: does it lift the exploit_never_proposed ceiling / goal-reach?

Both arms share proposer model+temperature, boxes, seeds, budget, mode=llm_only (so the only thing that
varies is the toggle under test). Saves per-step traces; reports consecutive_repeat_rate, wasted_rate,
exploit_proposed rate, goal-reach, each with an episode-clustered permutation test.

    STAGE2_LIVE_AUTHORIZED=... DEEPSEEK_API_KEY=... \
    python -m stage2.improvement_ab --compare memory --trials 6 --confirmed-isolated
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.engagement import run_engagement, LLMProposer, _make_executor  # noqa: E402
from stage2.eta import load_target  # noqa: E402
from stage2.safety import AuditLog  # noqa: E402
from stage2.stats_analysis import stratified_permutation, wilson  # noqa: E402

# exploit-proposable self-advertising boxes (where the ceiling / spinning are actually exercised)
DEFAULT_BOXES = [
    ("thinkphp-5-rce.json", "ThinkPHP-5-rce"),
    ("thinkphp-5023-rce.json", "ThinkPHP-5.0.23"),
    ("struts2-s2-048.json", "Struts2-S2-048"),
    ("struts2-s2-045.json", "Struts2-S2-045"),
    ("drupal-cve-2018-7600.json", "Drupalgeddon2"),
    ("joomla-cve-2017-8917.json", "Joomla-8917-sqli"),
    ("tomcat-cve-2017-12615.json", "Tomcat-12615"),
    ("php-cgi-cve-2012-1823.json", "php-cgi-2012-1823"),
]


def _consecutive_repeats(steps):
    acts = [s["chosen_action"] for s in steps]
    if len(acts) < 2:
        return 0, 0
    rep = sum(1 for i in range(1, len(acts)) if acts[i] == acts[i - 1])
    return rep, len(acts) - 1


def main():
    ap = argparse.ArgumentParser(description="A/B two engagement configs (memory or proposer).")
    ap.add_argument("--compare", choices=["memory", "proposer", "proposer_generic"], required=True,
                    help="proposer=baseline vs CVE-named enhanced prompt; proposer_generic=baseline vs "
                         "generic strong prompt (strategy only, NO test-set CVE names — leakage control)")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--mode", default="llm_only", choices=["llm_only", "prm"])
    ap.add_argument("--executor", choices=["dryrun", "live"], default="live")
    ap.add_argument("--confirmed-isolated", action="store_true")
    ap.add_argument("--boxes", nargs="+", default=None)
    ap.add_argument("--report-output", type=Path, default=None)
    args = ap.parse_args()

    if args.compare == "memory":
        arms = {"baseline_poormem": {"rich_memory": False, "variant": "baseline"},
                "rich_memory": {"rich_memory": True, "variant": "baseline"}}
    elif args.compare == "proposer":
        arms = {"baseline_prompt": {"rich_memory": False, "variant": "baseline"},
                "enhanced_prompt": {"rich_memory": False, "variant": "enhanced"}}
    else:  # proposer_generic — leakage control
        arms = {"baseline_prompt": {"rich_memory": False, "variant": "baseline"},
                "generic_prompt": {"rich_memory": False, "variant": "generic"}}
    out_path = args.report_output or (ROOT / "outputs" / f"stage2_improvement_{args.compare}.json")

    prm = joblib.load(ROOT / "outputs" / "prm_strong.joblib")
    audit = AuditLog(ROOT / "outputs" / "stage2_improvement_audit.jsonl")
    boxes = [(f, l) for f, l in DEFAULT_BOXES if (args.boxes is None or l in args.boxes)]

    episodes = {a: [] for a in arms}     # per-arm list of episode dicts (with steps)
    for fn, label in boxes:
        desc = load_target(ROOT / "stage2" / "targets" / fn)
        for seed in range(args.trials):
            for arm, cfg in arms.items():
                ex = _make_executor(args.executor, desc, audit, None, args.confirmed_isolated)
                prop = LLMProposer(model=args.model, temperature=args.temperature,
                                   prompt_variant=cfg["variant"])
                try:
                    r = run_engagement(desc, prm=prm, executor=ex, proposer=prop, mode=args.mode,
                                       budget=args.budget, audit=audit, permissive_guard=True,
                                       rerank_seed=seed, rich_memory=cfg["rich_memory"])
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:20s} seed={seed} {arm:18s} ERR {type(e).__name__}: {str(e)[:50]}", flush=True)
                    continue
                rep, den = _consecutive_repeats(r.get("steps", []))
                episodes[arm].append({"box": label, "arm": arm,
                                      "goal": int(bool(r["goal_reached"])),
                                      "exploit_proposed": int(bool(r.get("exploit_proposed"))),
                                      "prog_steps": r["progress_steps"], "total_steps": r["steps_taken"],
                                      "wasted": r["wasted_actions"], "rep": rep, "rep_den": den, "_one": 1})
        # progress line per box
        def _r(arm, key, den_key=None):
            es = [e for e in episodes[arm] if e["box"] == label]
            num = sum(e[key] for e in es)
            den = sum(e[den_key] for e in es) if den_key else len(es)
            return f"{num}/{den}={num/max(den,1):.0%}"
        a1, a2 = list(arms)
        print(f"{label:20s} {a1}: rep {_r(a1,'rep','rep_den')} wasted {_r(a1,'wasted','total_steps')} "
              f"goal {_r(a1,'goal')} | {a2}: rep {_r(a2,'rep','rep_den')} wasted {_r(a2,'wasted','total_steps')} "
              f"goal {_r(a2,'goal')}", flush=True)

    # pooled metrics + clustered permutation (treatment vs baseline)
    base, treat = list(arms)
    allep = episodes[base] + episodes[treat]
    # relabel arms to prm/llm_only so stratified_permutation (which keys on those) can be reused
    def perm(num_key, den_key):
        rows = [{**e, "arm": "prm" if e["arm"] == treat else "llm_only"} for e in allep]
        return stratified_permutation(rows, num_key, den_key)

    def summ(arm):
        es = episodes[arm]
        n = len(es)
        steps = sum(e["total_steps"] for e in es)
        rep = sum(e["rep"] for e in es); repden = sum(e["rep_den"] for e in es)
        wasted = sum(e["wasted"] for e in es)
        return {"n": n, "episodes_steps": steps,
                "consecutive_repeat_rate": round(rep / max(repden, 1), 3),
                "wasted_rate": round(wasted / max(steps, 1), 3),
                "exploit_proposed_rate": round(sum(e["exploit_proposed"] for e in es) / max(n, 1), 3),
                "goal_rate": round(sum(e["goal"] for e in es) / max(n, 1), 3),
                "mean_steps": round(steps / max(n, 1), 2)}

    obs_rep, p_rep = perm("rep", "rep_den")
    obs_wasted, p_wasted = perm("wasted", "total_steps")
    for e in allep:
        e["_one"] = 1
    obs_goal, p_goal = perm("goal", "_one")
    obs_exp, p_exp = perm("exploit_proposed", "_one")

    report = {
        "compare": args.compare, "model": args.model, "temperature": args.temperature,
        "mode": args.mode, "trials_per_box": args.trials, "boxes": [l for _, l in boxes],
        "baseline_arm": base, "treatment_arm": treat,
        base: summ(base), treat: summ(treat),
        "tests_treatment_vs_baseline": {
            "consecutive_repeat_rate": {"delta": round(obs_rep, 3), "perm_p": round(p_rep, 4)},
            "wasted_rate": {"delta": round(obs_wasted, 3), "perm_p": round(p_wasted, 4)},
            "goal_rate": {"delta": round(obs_goal, 3), "perm_p": round(p_goal, 4)},
            "exploit_proposed_rate": {"delta": round(obs_exp, 3), "perm_p": round(p_exp, 4)}},
        "episodes": [{k: e[k] for k in ("box", "arm", "goal", "exploit_proposed", "total_steps",
                                        "wasted", "rep", "rep_den")} for e in allep]}
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {args.compare} A/B: {base} vs {treat} (n={summ(base)['n']}/{summ(treat)['n']}) ===")
    for arm in (base, treat):
        s = summ(arm)
        print(f"  {arm:18s} repeat={s['consecutive_repeat_rate']:.0%} wasted={s['wasted_rate']:.0%} "
              f"exploit_proposed={s['exploit_proposed_rate']:.0%} goal={s['goal_rate']:.0%} steps={s['mean_steps']}")
    t = report["tests_treatment_vs_baseline"]
    print(f"  Δ(treat-base) clustered-permutation: repeat {t['consecutive_repeat_rate']['delta']:+} "
          f"p={t['consecutive_repeat_rate']['perm_p']} | wasted {t['wasted_rate']['delta']:+} "
          f"p={t['wasted_rate']['perm_p']} | exploit_proposed {t['exploit_proposed_rate']['delta']:+} "
          f"p={t['exploit_proposed_rate']['perm_p']} | goal {t['goal_rate']['delta']:+} p={t['goal_rate']['perm_p']}")
    print(f"report -> {out_path.name}")


if __name__ == "__main__":
    main()
