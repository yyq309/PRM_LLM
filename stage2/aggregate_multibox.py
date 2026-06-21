"""Aggregate the per-box live A/B trial reports into the multi-box uplift result.

Reads each stage2_ab_trials_*.json, prints per-box goal-reach (PRM vs llm_only), then the POOLED
result over the self-advertising boxes (php-cgi, whose vuln is not fingerprint-identifiable, is
reported separately because the loop fails on both arms there). Includes Wilson CIs and a
two-proportion z-test on the pooled difference, so the (under)powered nature is explicit.

    python -m stage2.aggregate_multibox
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]

# (report file, short label, self-advertising vuln?)
BOXES = [
    ("stage2_live_ab_trials.json", "ThinkPHP-5-rce", True),
    ("stage2_ab_trials_tp5023.json", "ThinkPHP-5.0.23", True),
    ("stage2_ab_trials_struts2.json", "Struts2-S2-048", True),
    ("stage2_ab_trials_struts2_045.json", "Struts2-S2-045", True),
    ("stage2_ab_trials_drupal.json", "Drupalgeddon2", True),
    ("stage2_ab_trials_tomcat.json", "Tomcat-12615", True),
    ("stage2_ab_trials_joomla.json", "Joomla-8917-sqli", True),
    ("stage2_ab_trials_phpcgi.json", "php-cgi-2012-1823", False),
    ("stage2_ab_trials_phpinc.json", "php-inclusion-LFI", False),
    ("stage2_ab_trials_rails.json", "Rails-5418-fileread", False),
    ("stage2_ab_trials_weblogic.json", "WebLogic-weakpw(auth)", False),
    ("stage2_ab_trials_gitea.json", "Gitea-1.4(auth)", False),
    ("stage2_ab_trials_tomcat8.json", "Tomcat8-weakpw", True),
    ("stage2_ab_trials_httpd.json", "httpd-41773", True),
    ("stage2_ab_trials_nginx.json", "nginx-insecure", True),
]


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return None, None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided p via erfc
    pval = math.erfc(abs(z) / math.sqrt(2))
    return round(z, 3), round(pval, 3)


def _load(name):
    p = ROOT / "outputs" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    rows = []
    pool = {"adv": {"prm": [0, 0], "llm": [0, 0]}, "all": {"prm": [0, 0], "llm": [0, 0]}}
    # high-N per-step-progress pool (the powered metric, vs the underpowered per-episode goal)
    step_pool = {"adv": {"prm": [0, 0], "llm": [0, 0]}, "all": {"prm": [0, 0], "llm": [0, 0]}}
    for fn, label, adv in BOXES:
        d = _load(fn)
        if d is None:
            print(f"  (missing {fn})")
            continue
        pk, pn = d["prm"]["goal_reached"], d["prm"]["n"]
        lk, ln = d["llm_only"]["goal_reached"], d["llm_only"]["n"]
        rows.append((label, adv, pk, pn, lk, ln))
        pool["all"]["prm"][0] += pk; pool["all"]["prm"][1] += pn
        pool["all"]["llm"][0] += lk; pool["all"]["llm"][1] += ln
        if adv:
            pool["adv"]["prm"][0] += pk; pool["adv"]["prm"][1] += pn
            pool["adv"]["llm"][0] += lk; pool["adv"]["llm"][1] += ln
        for arm, key in (("prm", "prm"), ("llm", "llm_only")):
            sp = d.get(key, {}).get("per_step_progress")
            if sp:
                step_pool["all"][arm][0] += sp["progress_steps"]; step_pool["all"][arm][1] += sp["total_steps"]
                if adv:
                    step_pool["adv"][arm][0] += sp["progress_steps"]; step_pool["adv"][arm][1] += sp["total_steps"]

    print("Per-box goal-reach (PRM vs llm_only):\n")
    print(f"  {'box':20s} {'adv':4s} {'PRM':>10s}  {'llm_only':>10s}")
    for label, adv, pk, pn, lk, ln in rows:
        print(f"  {label:20s} {'yes' if adv else 'no':4s} "
              f"{pk}/{pn}={pk/max(pn,1):.0%}".rjust(10) + "  " +
              f"{lk}/{ln}={lk/max(ln,1):.0%}".rjust(10))

    out = {"per_box": [{"box": l, "self_advertising": a, "prm_goal": pk, "prm_n": pn,
                        "llm_only_goal": lk, "llm_only_n": ln} for l, a, pk, pn, lk, ln in rows]}
    for key, title in [("adv", "POOLED over self-advertising boxes"), ("all", "POOLED over ALL boxes")]:
        pk, pn = pool[key]["prm"]; lk, ln = pool[key]["llm"]
        plo, phi = wilson(pk, pn); llo, lhi = wilson(lk, ln)
        z, pval = two_prop_z(pk, pn, lk, ln)
        print(f"\n{title}:")
        print(f"  PRM      {pk}/{pn} = {pk/max(pn,1):.1%}  CI95[{plo:.2f},{phi:.2f}]")
        print(f"  llm_only {lk}/{ln} = {lk/max(ln,1):.1%}  CI95[{llo:.2f},{lhi:.2f}]")
        print(f"  Δ = {(pk/max(pn,1)-lk/max(ln,1))*100:+.1f}pp   two-proportion z={z}, p={pval} "
              f"({'NOT significant' if (pval or 1) >= 0.05 else 'significant'} at 0.05)")
        out[key] = {"prm": [pk, pn], "llm_only": [lk, ln],
                    "prm_rate": round(pk / max(pn, 1), 3), "llm_only_rate": round(lk / max(ln, 1), 3),
                    "delta_pp": round((pk / max(pn, 1) - lk / max(ln, 1)) * 100, 1),
                    "prm_ci95": [round(plo, 3), round(phi, 3)], "llm_only_ci95": [round(llo, 3), round(lhi, 3)],
                    "two_prop_z": z, "two_prop_p": pval}

    # --- the POWERED metric: per-step progress pooled over all steps (high N) ---
    print("\nPER-STEP PROGRESS (high-N; the sensitive metric, not per-episode success):")
    for key, title in [("adv", "self-advertising boxes"), ("all", "all boxes")]:
        pk, pn = step_pool[key]["prm"]; lk, ln = step_pool[key]["llm"]
        if pn == 0:
            print(f"  {title}: (no per-step data — re-run with the instrumented engagement)")
            continue
        plo, phi = wilson(pk, pn); llo, lhi = wilson(lk, ln)
        z, pval = two_prop_z(pk, pn, lk, ln)
        print(f"  [{title}] PRM {pk}/{pn}={pk/pn:.1%} CI[{plo:.2f},{phi:.2f}]  "
              f"vs llm_only {lk}/{ln}={lk/ln:.1%} CI[{llo:.2f},{lhi:.2f}]  "
              f"Δ={(pk/pn-lk/ln)*100:+.1f}pp  z={z} p={pval} "
              f"({'NOT sig' if (pval or 1) >= 0.05 else 'SIGNIFICANT'})")
        out["per_step_progress_" + key] = {"prm": [pk, pn], "llm_only": [lk, ln],
            "prm_rate": round(pk / pn, 3), "llm_only_rate": round(lk / ln, 3),
            "delta_pp": round((pk / pn - lk / ln) * 100, 1), "two_prop_z": z, "two_prop_p": pval}

    (ROOT / "outputs" / "stage2_multibox_aggregate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> outputs/stage2_multibox_aggregate.json")


if __name__ == "__main__":
    main()
