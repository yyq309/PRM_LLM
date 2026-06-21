"""Emit the FINAL Stage-2 7-dimension per-box table (all boxes) + a synthesis, straight from the
trial JSONs, the cluster-robust stats (stage2_stats_analysis.json), and the reranker-isolation
ablation (stage2_ablation_rerank.json). Single source of truth for STAGE2_LIVE_RESULTS.md and the
report handed to the user — no hand-transcribed numbers.

    python -m stage2.seven_dim_report          # prints table + writes STAGE2_SEVEN_DIM_TABLE.md
"""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

# (trial json, label, vuln class, self-adv, full-goal-reachable)
BOXES = [
    ("stage2_live_ab_trials.json",      "ThinkPHP-5-rce",       "RCE",  True,  True),
    ("stage2_ab_trials_tp5023.json",    "ThinkPHP-5.0.23",      "RCE",  True,  True),
    ("stage2_ab_trials_struts2.json",   "Struts2-S2-048",       "RCE",  True,  True),
    ("stage2_ab_trials_struts2_045.json","Struts2-S2-045",      "RCE",  True,  True),
    ("stage2_ab_trials_drupal.json",    "Drupalgeddon2",        "RCE",  True,  True),
    ("stage2_ab_trials_tomcat.json",    "Tomcat-12615",         "RCE",  True,  True),
    ("stage2_ab_trials_joomla.json",    "Joomla-8917-sqli",     "SQLi", True,  True),
    ("stage2_ab_trials_phpcgi.json",    "php-cgi-2012-1823",    "RCE",  False, True),
    ("stage2_ab_trials_phpinc.json",    "php-inclusion-LFI",    "LFI",  False, True),
    ("stage2_ab_trials_rails.json",     "Rails-5418-fileread",  "LFI",  False, True),
    ("stage2_ab_trials_weblogic.json",  "WebLogic-weakpw",      "auth", False, False),
    ("stage2_ab_trials_gitea.json",     "Gitea-1.4",            "auth", False, False),
    ("stage2_ab_trials_tomcat8.json",   "Tomcat8-weakpw",       "RCE",  True,  True),
    ("stage2_ab_trials_httpd.json",     "httpd-41773",          "RCE",  True,  True),
    ("stage2_ab_trials_nginx.json",     "nginx-insecure",       "LFI",  True,  False),
]


def _load(name):
    p = ROOT / "outputs" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    stats = _load("stage2_stats_analysis.json") or {}
    perbox_stats = {b["box"]: b for b in stats.get("per_box", [])}
    lines = []
    lines.append("# Stage-2 inference — FINAL 7-dimension table (all 12 live boxes)\n")
    lines.append("PRM-rerank vs llm_only (proposer order). 5 trials/arm, deepseek-chat proposer, live "
                 "gated execution. `goal*` columns use the milestone-robust goal (cmd-exec ∧ file-read, "
                 "or root/flag). Auth-milestone boxes (WebLogic, Gitea) have no full goal by design.\n")
    # 7-dim header
    hdr = ("| box | class | adv | step-progress PRM | step-progress llm | goal PRM | goal llm | "
           "shell/cmd/file PRM | wasted PRM/llm | calls·exec PRM | live-OOA PRM | gate-ref |")
    sep = "|" + "---|" * 12
    lines.append(hdr); lines.append(sep)

    pool = {"prm": {"ps": [0, 0], "g": [0, 0]}, "llm": {"ps": [0, 0], "g": [0, 0]}}
    for fn, label, vclass, adv, reachable in BOXES:
        d = _load(fn)
        if not d:
            lines.append(f"| {label} | {vclass} | — | (missing) |")
            continue
        prm, llm = d["prm"], d["llm_only"]
        ps_p = prm["per_step_progress"]; ps_l = llm["per_step_progress"]
        gp = f"{prm['goal_reached']}/{prm['n']}"
        gl = f"{llm['goal_reached']}/{llm['n']}"
        sb = f"{prm['shell_rate']:.0%}/{prm['command_exec_rate']:.0%}/{prm['file_read_rate']:.0%}"
        wasted = f"{prm['mean_wasted_rate']:.0%}/{llm['mean_wasted_rate']:.0%}"
        cost = f"{prm['mean_proposer_calls']:.0f}·{prm['mean_eta_executions']:.0f}"
        ooa = f"{prm['mean_live_out_of_abstraction_rate']:.0%}"
        gate = prm["total_gate_refusals"] + llm["total_gate_refusals"]
        lines.append(
            f"| {label} | {vclass} | {'Y' if adv else 'n'} | "
            f"{ps_p['rate']:.0%} ({ps_p['progress_steps']}/{ps_p['total_steps']}) | "
            f"{ps_l['rate']:.0%} ({ps_l['progress_steps']}/{ps_l['total_steps']}) | "
            f"{gp} | {gl} | {sb} | {wasted} | {cost} | {ooa} | {gate} |")
        pool["prm"]["ps"][0] += ps_p["progress_steps"]; pool["prm"]["ps"][1] += ps_p["total_steps"]
        pool["llm"]["ps"][0] += ps_l["progress_steps"]; pool["llm"]["ps"][1] += ps_l["total_steps"]
        if reachable:
            pool["prm"]["g"][0] += prm["goal_reached"]; pool["prm"]["g"][1] += prm["n"]
            pool["llm"]["g"][0] += llm["goal_reached"]; pool["llm"]["g"][1] += llm["n"]

    pp = pool["prm"]; pl = pool["llm"]
    lines.append("")
    lines.append(f"**Pooled per-step progress:** PRM {pp['ps'][0]}/{pp['ps'][1]}"
                 f"={pp['ps'][0]/max(pp['ps'][1],1):.1%} vs llm_only {pl['ps'][0]}/{pl['ps'][1]}"
                 f"={pl['ps'][0]/max(pl['ps'][1],1):.1%}.  "
                 f"**Pooled full-goal (reachable boxes):** PRM {pp['g'][0]}/{pp['g'][1]}"
                 f"={pp['g'][0]/max(pp['g'][1],1):.0%} vs llm_only {pl['g'][0]}/{pl['g'][1]}"
                 f"={pl['g'][0]/max(pl['g'][1],1):.0%}.")

    # cluster-robust headline pulled from stats_analysis.json
    if stats:
        sa = stats["pooled"]["all_full"]["per_step"]
        se = stats["pooled"]["all_full"]["per_episode"]
        ga = stats["progress_variants"]["goal_aligned"]
        lines.append("")
        lines.append("## Cluster-robust significance (episode-clustered permutation + cluster bootstrap)\n")
        lines.append("| metric (ALL full-goal boxes) | Δ | naive z p | **permutation p (clustered)** | cluster-boot CI95(Δ) | verdict |")
        lines.append("|---|--:|--:|--:|--:|---|")
        for tag, b in [("per-step progress", sa), ("goal-aligned progress (forward-action)", ga),
                       ("per-episode goal", se)]:
            es = b["effect_size"]
            lines.append(f"| {tag} | {es['risk_diff_pp']:+}pp | {b['naive_p']} | "
                         f"**{b['perm_p_clustered']}** | "
                         f"[{b['cluster_boot_ci95_diff_pp'][0]},{b['cluster_boot_ci95_diff_pp'][1]}]pp | "
                         f"{'SIGNIFICANT' if b['significant_clustered'] else 'NS'} |")
        # failure taxonomy
        tax = stats["failure_taxonomy"]
        lines.append("\n## Failure taxonomy (per-episode terminal reason, by arm)\n")
        lines.append("| reason | PRM | llm_only |\n|---|--:|--:|")
        for k in ["success", "foothold_no_file", "exploit_executed_no_foothold", "exploit_never_proposed",
                  "budget_exhausted", "goal_unreachable_by_design", "safety_refusal"]:
            lines.append(f"| {k} | {tax['prm'].get(k,0)} | {tax['llm_only'].get(k,0)} |")

    # reranker-isolation ablations (deterministic + LLM proposer)
    for fn_abl, title in [("stage2_ablation_rerank.json", "deterministic proposer, key-free (full action surface)"),
                          ("stage2_ablation_rerank_llm.json", "REAL LLM proposer (targeted candidates)")]:
        abl = _load(fn_abl)
        if not abl:
            continue
        lines.append(f"\n## Reranker-isolation ablation — {title}\n")
        lines.append(f"Proposer: `{abl['meta']['proposer']}`; only the rerank function varies. "
                     f"Pooled over full-goal boxes ({', '.join(abl['meta'].get('boxes', [])[:8])}):\n")
        lines.append("| rerank mode | per-step progress | goal-reach |\n|---|--:|--:|")
        pf = abl["pooled_full_goal"]
        for m in abl["meta"]["modes"]:
            r = pf[m]
            star = " **(PRM)**" if m == "prm" else ""
            lines.append(f"| {m}{star} | {r['progress_rate']:.1%} ({r['progress'][0]}/{r['progress'][1]}) | "
                         f"{r['goal_rate']:.0%} ({r['goal'][0]}/{r['goal'][1]}) |")
        lines.append("\n**prm vs each baseline (episode-clustered permutation, per-step progress):**\n")
        for k, v in abl["permutation_prm_vs_baseline"].items():
            lines.append(f"- `{k}`: Δ={v['delta_pp']:+}pp, perm-p={v['perm_p_clustered']} "
                         f"({'SIGNIFICANT' if v['significant'] else 'NS'})")

    text = "\n".join(lines) + "\n"
    (ROOT / "STAGE2_SEVEN_DIM_TABLE.md").write_text(text, encoding="utf-8")
    print(text)
    print("written -> STAGE2_SEVEN_DIM_TABLE.md")


if __name__ == "__main__":
    main()
