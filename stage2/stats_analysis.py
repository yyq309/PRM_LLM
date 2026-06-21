"""Cluster-robust statistical re-analysis of the Stage-2 live A/B per-box trial reports.

A naive two-proportion z-test on POOLED per-step progress treats every step as an independent
Bernoulli trial. It is not: steps within an episode (and episodes within a box) are correlated, so
the naive test UNDER-counts the variance and OVER-states significance. This module re-tests the same
data with cluster-aware methods and reports the honest picture:

  * stratified PERMUTATION test  -- randomization unit = the whole episode (a cluster of steps),
    permuted within each box (box as a blocking stratum). Assumption-light gold standard.
  * CLUSTER BOOTSTRAP CIs         -- resample whole episodes within box with replacement (not steps).
  * GEE (statsmodels)            -- binomial GEE, exchangeable working corr, clustered by episode
    (per-step) / by box (per-episode); the Wald p on the arm term is cluster-robust.
  * effect sizes                 -- risk difference, risk ratio, odds ratio, Cohen's h.
  * per-box + pooled + stratified-by-class, each with CI + p + effect size + significance flag.
  * auth-milestone boxes reported SEPARATELY (their goal is unreachable by design -> excluded from
    full-goal denominators, reported on the milestone metric instead).
  * goal-aligned / weighted / milestone progress variants (stricter than "any state change").
  * a failure-mode taxonomy.

All randomness uses a FIXED seed (recorded in the output metadata) so the report is reproducible.

    python -m stage2.stats_analysis
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # GEE iteration-limit / overflow noise on tiny strata

ROOT = Path(__file__).resolve().parents[1]
SEED = 12345
N_PERM = 20000
N_BOOT = 10000

# Recon / non-forward actions: making the abstract state change here is NOT goal-aligned progress.
RECON_ACTIONS = {
    "http_fingerprint", "content_retrieval", "web_path_enumeration",
    "service_enumeration", "input_discovery", "form_interaction",
}
# Forward actions: progress on one of these moves toward exploitation / the goal.
FORWARD_ACTIONS = {
    "vulnerability_check", "exploit_attempt", "command_execution",
    "sensitive_file_read", "privilege_escalation", "file_upload_attempt", "auth_attempt",
}
# Action weights for the weighted_progress_score (recon cheap, foothold/read/root expensive).
ACTION_WEIGHT = {
    "http_fingerprint": 0.2, "content_retrieval": 0.2, "web_path_enumeration": 0.3,
    "service_enumeration": 0.3, "input_discovery": 0.3, "form_interaction": 0.3,
    "vulnerability_check": 0.6, "auth_attempt": 0.8, "exploit_attempt": 1.0,
    "file_upload_attempt": 1.0, "command_execution": 1.2, "sensitive_file_read": 1.5,
    "privilege_escalation": 2.0,
}
# Milestone ladder for milestone_progress (max level reached, normalised by 5).
MILESTONE_LADDER = 5  # recon=0 < vuln=1 < shell=2 < cmd=3 < file=4 < root=5

# (report file, label, vuln_class, self_advertising, full_goal_reachable_by_design)
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
    # --- 3 new boxes (2026-06-21), run with budget 16 / patience 6 for the multi-step chains ---
    ("stage2_ab_trials_tomcat8.json",   "Tomcat8-weakpw",       "RCE",  True,  True),
    ("stage2_ab_trials_httpd.json",     "httpd-41773",          "RCE",  True,  True),
    ("stage2_ab_trials_nginx.json",     "nginx-insecure",       "LFI",  True,  False),
]


# --------------------------------------------------------------------------- loaders
def _classify_failure(ep, goal_reachable):
    if ep.get("goal_reached"):
        return "success"
    if not goal_reachable:
        return "goal_unreachable_by_design"
    if ep.get("gate_refusals", 0) and ep.get("steps_taken", 0) == 0:
        return "safety_refusal"
    if ep.get("stop_reason") == "budget_exhausted":
        return "budget_exhausted"
    acts = [s.get("chosen_action") for s in ep.get("steps", []) if s.get("executed")]
    exploit_acts = {"exploit_attempt", "file_upload_attempt", "auth_attempt", "command_execution"}
    if not (set(acts) & exploit_acts):
        return "exploit_never_proposed"          # proposer never surfaced an exploit
    if ep.get("reached_command_exec") or ep.get("reached_shell"):
        return "foothold_no_file"                 # got a foothold, never read the file (ranking/tail)
    return "exploit_executed_no_foothold"         # exploit ran but no shell (eta-recipe / phi-parse / target)


def _milestone_level(ep):
    if ep.get("reached_root"):
        return 5
    if ep.get("read_any_file"):
        return 4
    if ep.get("reached_command_exec"):
        return 3
    if ep.get("reached_shell"):
        return 2
    acts = {s.get("chosen_action") for s in ep.get("steps", []) if s.get("executed")}
    if acts & {"vulnerability_check", "exploit_attempt", "auth_attempt", "file_upload_attempt"}:
        return 1
    return 0


def load_episodes():
    """Flatten every per-episode record across boxes into a single list of dicts."""
    eps = []
    for fn, label, vclass, self_adv, reachable in BOXES:
        p = ROOT / "outputs" / fn
        if not p.exists():
            print(f"  (missing {fn})", file=sys.stderr)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for arm in ("prm", "llm_only"):
            for ep in d.get("per_trial", {}).get(arm, []):
                steps = ep.get("steps", [])
                prog = sum(1 for s in steps if s.get("made_progress"))
                tot = len(steps)
                # goal-aligned: progress on a FORWARD action only
                ga = sum(1 for s in steps if s.get("made_progress")
                         and s.get("chosen_action") in FORWARD_ACTIONS)
                wsum = sum(ACTION_WEIGHT.get(s.get("chosen_action"), 0.5)
                           for s in steps if s.get("made_progress"))
                wmax = sum(ACTION_WEIGHT.get(s.get("chosen_action"), 0.5) for s in steps) or 1.0
                eps.append({
                    "box": label, "vclass": vclass, "self_adv": self_adv, "reachable": reachable,
                    "arm": arm,
                    "goal": int(bool(ep.get("goal_reached"))),
                    "prog_steps": prog, "total_steps": tot,
                    "ga_steps": ga,
                    "weighted_progress": wsum / wmax if tot else 0.0,
                    "milestone": _milestone_level(ep),
                    "shell": int(bool(ep.get("reached_shell"))),
                    "cmd": int(bool(ep.get("reached_command_exec"))),
                    "file": int(bool(ep.get("read_any_file"))),
                    "root": int(bool(ep.get("reached_root"))),
                    "failure": _classify_failure(ep, reachable),
                    "proposer_calls": ep.get("proposer_calls", 0),
                    "eta_execs": ep.get("eta_executions", 0),
                    "gate_refusals": ep.get("gate_refusals", 0),
                    "live_ooa": ep.get("live_out_of_abstraction_rate", 0.0),
                    "wasted_rate": ep.get("wasted_rate", 0.0),
                })
    return eps


# --------------------------------------------------------------------------- basic stats
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
    return round(z, 3), round(math.erfc(abs(z) / math.sqrt(2)), 4)


def cohen_h(p1, p2):
    return round(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)), 3)


def effect_sizes(k1, n1, k2, n2):
    """k1/n1 = prm, k2/n2 = llm_only. RD, RR, OR, Cohen's h."""
    if n1 == 0 or n2 == 0:
        return {}
    p1, p2 = k1 / n1, k2 / n2
    rr = (p1 / p2) if p2 > 0 else None
    # Haldane-Anscombe 0.5 correction for OR when a cell is 0
    a, b, c, d = k1 + 0.5, n1 - k1 + 0.5, k2 + 0.5, n2 - k2 + 0.5
    orr = (a * d) / (b * c)
    return {
        "risk_diff_pp": round((p1 - p2) * 100, 1),
        "risk_ratio": round(rr, 2) if rr is not None else None,
        "odds_ratio": round(orr, 2),
        "cohen_h": cohen_h(p1, p2),
    }


# --------------------------------------------------------------------------- cluster-robust tests
def _pooled_rate_diff(rows, num_key, den_key):
    """prm pooled rate - llm pooled rate over a list of episode rows."""
    pk = sum(r[num_key] for r in rows if r["arm"] == "prm")
    pn = sum(r[den_key] for r in rows if r["arm"] == "prm")
    lk = sum(r[num_key] for r in rows if r["arm"] == "llm_only")
    ln = sum(r[den_key] for r in rows if r["arm"] == "llm_only")
    rp = pk / pn if pn else 0.0
    rl = lk / ln if ln else 0.0
    return rp - rl, (pk, pn, lk, ln)


def stratified_permutation(rows, num_key, den_key, n_perm=N_PERM, seed=SEED):
    """Permute the ARM label of whole episodes within each box (box = stratum).

    Steps move with their episode (the cluster), so within-episode correlation is preserved under H0.
    Returns two-sided p for the pooled rate difference.
    """
    rng = np.random.default_rng(seed)
    obs, _ = _pooled_rate_diff(rows, num_key, den_key)
    # group indices by box
    by_box = {}
    for i, r in enumerate(rows):
        by_box.setdefault(r["box"], []).append(i)
    arms = np.array([r["arm"] for r in rows], dtype=object)
    num = np.array([r[num_key] for r in rows], dtype=float)
    den = np.array([r[den_key] for r in rows], dtype=float)
    ge = 0
    for _ in range(n_perm):
        perm = arms.copy()
        for box, idx in by_box.items():
            sub = perm[idx].copy()
            rng.shuffle(sub)
            perm[idx] = sub
        mp = perm == "prm"
        ml = perm == "llm_only"
        pn, ln = den[mp].sum(), den[ml].sum()
        rp = num[mp].sum() / pn if pn else 0.0
        rl = num[ml].sum() / ln if ln else 0.0
        if abs(rp - rl) >= abs(obs) - 1e-12:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)  # add-one (never reports p=0)


def cluster_bootstrap_ci(rows, num_key, den_key, n_boot=N_BOOT, seed=SEED):
    """Resample whole episodes (clusters) WITH replacement within each box; percentile CI of diff."""
    rng = np.random.default_rng(seed + 1)
    by_box_arm = {}
    for r in rows:
        by_box_arm.setdefault((r["box"], r["arm"]), []).append(r)
    diffs = []
    for _ in range(n_boot):
        pk = pn = lk = ln = 0.0
        for (box, arm), lst in by_box_arm.items():
            m = len(lst)
            pick = rng.integers(0, m, m)
            for j in pick:
                r = lst[j]
                if arm == "prm":
                    pk += r[num_key]; pn += r[den_key]
                else:
                    lk += r[num_key]; ln += r[den_key]
        rp = pk / pn if pn else 0.0
        rl = lk / ln if ln else 0.0
        diffs.append(rp - rl)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo) * 100, 1), round(float(hi) * 100, 1)


def gee_pvalue(rows, kind):
    """Binomial GEE; Wald p on the arm coefficient, cluster-robust.

    kind='per_step'  -> one row per step, y=made_progress,  group=episode
    kind='per_episode'-> one row per episode, y=goal,        group=box
    Returns (coef, p) or (None, None) if it fails to fit.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.genmod.cov_struct as cov
        import pandas as pd
    except Exception:
        return None, None
    recs = []
    if kind == "per_step":
        for ei, r in enumerate(rows):
            # reconstruct step-level rows from counts (progress vs not) within the episode cluster
            for _ in range(r["prog_steps"]):
                recs.append({"y": 1, "arm": 1 if r["arm"] == "prm" else 0, "grp": ei})
            for _ in range(r["total_steps"] - r["prog_steps"]):
                recs.append({"y": 0, "arm": 1 if r["arm"] == "prm" else 0, "grp": ei})
    else:
        for r in rows:
            recs.append({"y": r["goal"], "arm": 1 if r["arm"] == "prm" else 0, "grp": r["box"]})
    if not recs:
        return None, None
    df = pd.DataFrame(recs)
    if df["y"].nunique() < 2:
        return None, None
    df["grp"] = pd.factorize(df["grp"])[0]
    if df["grp"].nunique() < 3:
        return None, None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.GEE.from_formula("y ~ arm", groups="grp", data=df,
                                    family=sm.families.Binomial(), cov_struct=cov.Exchangeable())
            res = m.fit(maxiter=200)
        coef = float(res.params["arm"]); pv = float(res.pvalues["arm"])
        # guard against non-convergence / quasi-separation (huge coef, p pinned to 0, non-finite)
        if not math.isfinite(coef) or not math.isfinite(pv) or abs(coef) > 8 or pv == 0.0:
            return None, None
        return round(coef, 3), round(pv, 4)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- reporting blocks
def per_episode_block(rows, title):
    pk = sum(r["goal"] for r in rows if r["arm"] == "prm")
    pn = sum(1 for r in rows if r["arm"] == "prm")
    lk = sum(r["goal"] for r in rows if r["arm"] == "llm_only")
    ln = sum(1 for r in rows if r["arm"] == "llm_only")
    z, p_naive = two_prop_z(pk, pn, lk, ln)
    _, p_perm = stratified_permutation(rows, "goal", None, ) if False else (None, None)
    # per-episode permutation: num=goal, den=1 (each episode contributes 1 to denominator)
    for r in rows:
        r["_one"] = 1
    obs, p_perm = stratified_permutation(rows, "goal", "_one")
    ci_lo, ci_hi = cluster_bootstrap_ci(rows, "goal", "_one")
    gcoef, gp = gee_pvalue(rows, "per_episode")
    es = effect_sizes(pk, pn, lk, ln)
    plo, phi = wilson(pk, pn); llo, lhi = wilson(lk, ln)
    return {
        "title": title, "metric": "per_episode_goal",
        "prm": [pk, pn], "llm_only": [lk, ln],
        "prm_rate": round(pk / pn, 3) if pn else None, "prm_ci95": [round(plo, 3), round(phi, 3)],
        "llm_only_rate": round(lk / ln, 3) if ln else None, "llm_only_ci95": [round(llo, 3), round(lhi, 3)],
        "naive_z": z, "naive_p": p_naive,
        "perm_p_clustered": round(p_perm, 4),
        "cluster_boot_ci95_diff_pp": [ci_lo, ci_hi],
        "gee_p_clustered_by_box": gp,
        "effect_size": es,
        "significant_clustered": bool(p_perm < 0.05),
    }


def per_step_block(rows, title, num_key="prog_steps", metric="per_step_progress"):
    diff, (pk, pn, lk, ln) = _pooled_rate_diff(rows, num_key, "total_steps")
    z, p_naive = two_prop_z(pk, pn, lk, ln)
    obs, p_perm = stratified_permutation(rows, num_key, "total_steps")
    ci_lo, ci_hi = cluster_bootstrap_ci(rows, num_key, "total_steps")
    gcoef, gp = (None, None)
    if num_key == "prog_steps":
        gcoef, gp = gee_pvalue(rows, "per_step")
    es = effect_sizes(pk, pn, lk, ln)
    plo, phi = wilson(pk, pn); llo, lhi = wilson(lk, ln)
    return {
        "title": title, "metric": metric,
        "prm": [pk, pn], "llm_only": [lk, ln],
        "prm_rate": round(pk / pn, 3) if pn else None, "prm_ci95": [round(plo, 3), round(phi, 3)],
        "llm_only_rate": round(lk / ln, 3) if ln else None, "llm_only_ci95": [round(llo, 3), round(lhi, 3)],
        "naive_z": z, "naive_p": p_naive,
        "perm_p_clustered": round(p_perm, 4),
        "cluster_boot_ci95_diff_pp": [ci_lo, ci_hi],
        "gee_p_clustered_by_episode": gp,
        "effect_size": es,
        "significant_clustered": bool(p_perm < 0.05),
    }


def fmt_block(b):
    pk, pn = b["prm"]; lk, ln = b["llm_only"]
    es = b["effect_size"]
    gee_key = "gee_p_clustered_by_episode" if "gee_p_clustered_by_episode" in b else "gee_p_clustered_by_box"
    return (
        f"  [{b['title']}] {b['metric']}\n"
        f"     PRM {pk}/{pn}={pk/max(pn,1):.1%} CI[{b['prm_ci95'][0]:.2f},{b['prm_ci95'][1]:.2f}]  "
        f"llm_only {lk}/{ln}={lk/max(ln,1):.1%} CI[{b['llm_only_ci95'][0]:.2f},{b['llm_only_ci95'][1]:.2f}]\n"
        f"     Δ={es.get('risk_diff_pp')}pp  RR={es.get('risk_ratio')}  OR={es.get('odds_ratio')}  h={es.get('cohen_h')}\n"
        f"     naive z-test p={b['naive_p']}  |  PERMUTATION p(clustered)={b['perm_p_clustered']}  "
        f"GEE p={b.get(gee_key)}  cluster-boot CI95(Δ)=[{b['cluster_boot_ci95_diff_pp'][0]},{b['cluster_boot_ci95_diff_pp'][1]}]pp\n"
        f"     -> {'SIGNIFICANT' if b['significant_clustered'] else 'NOT significant'} (clustered)\n"
    )


def holm_bonferroni(pairs, alpha=0.05):
    """Holm-Bonferroni step-down family-wise correction. `pairs` = list of (name, raw_p).
    Returns list of {name, raw_p, adj_p, significant, suggestive} sorted by raw_p. Controls the
    family-wise error rate across the ~40 clustered tests so a reviewer can't say we p-hacked."""
    m = len(pairs)
    ordered = sorted(pairs, key=lambda x: x[1])
    results, running = [], 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)          # step-down: i-th smallest scaled by (m-i)
        running = max(running, adj)          # enforce monotonic non-decreasing adjusted p
        results.append({"test": name, "raw_p": round(p, 4), "adj_p_holm": round(running, 4),
                        "significant": running < alpha, "suggestive": (p < alpha) and (running >= alpha)})
    return results


def _collect_pvalues(out):
    """Gather every clustered permutation p-value in the report into the correction family."""
    fam = []
    for b in out["per_box"]:
        fam.append((f"{b['box']}:per_episode_goal", b["per_episode"]["perm_p_clustered"]))
        fam.append((f"{b['box']}:per_step", b["per_step"]["perm_p_clustered"]))
    for key, blk in out["pooled"].items():
        fam.append((f"pooled[{key}]:per_episode_goal", blk["per_episode"]["perm_p_clustered"]))
        fam.append((f"pooled[{key}]:per_step", blk["per_step"]["perm_p_clustered"]))
    for vclass, blk in out["stratified"].items():
        fam.append((f"stratified[{vclass}]:per_step", blk["perm_p_clustered"]))
    if "goal_aligned" in out.get("progress_variants", {}):
        fam.append(("goal_aligned:per_step", out["progress_variants"]["goal_aligned"]["perm_p_clustered"]))
    return fam


def main():
    eps = load_episodes()
    full = [e for e in eps if e["reachable"]]            # exclude auth-milestone from goal denominators
    self_adv_full = [e for e in full if e["self_adv"]]
    auth = [e for e in eps if not e["reachable"]]
    out = {"meta": {"seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT,
                    "n_episodes": len(eps), "n_full_goal_episodes": len(full),
                    "n_auth_milestone_episodes": len(auth)},
           "per_box": [], "pooled": {}, "stratified": {}, "failure_taxonomy": {},
           "progress_variants": {}}

    # ---- per-box (full-goal boxes): per-episode goal + per-step progress ----
    print("=" * 96)
    print("PER-BOX (per-episode goal-reach + per-step progress), clustered tests")
    print("=" * 96)
    for fn, label, vclass, self_adv, reachable in BOXES:
        rows = [e for e in eps if e["box"] == label]
        if not rows:
            continue
        ep_b = per_episode_block(rows, label)
        st_b = per_step_block(rows, label)
        out["per_box"].append({"box": label, "vclass": vclass, "self_adv": self_adv,
                               "reachable": reachable, "per_episode": ep_b, "per_step": st_b})
        pk, pn = ep_b["prm"]; lk, ln = ep_b["llm_only"]
        spk, spn = st_b["prm"]; slk, sln = st_b["llm_only"]
        print(f"\n{label} [{vclass}{'/self-adv' if self_adv else ''}{'' if reachable else '/AUTH-MILESTONE'}]")
        print(f"   goal: PRM {pk}/{pn}={pk/max(pn,1):.0%} vs {lk}/{ln}={lk/max(ln,1):.0%}  "
              f"Δ={ep_b['effect_size'].get('risk_diff_pp')}pp perm-p={ep_b['perm_p_clustered']}")
        print(f"   step: PRM {spk}/{spn}={spk/max(spn,1):.0%} vs {slk}/{sln}={slk/max(sln,1):.0%}  "
              f"Δ={st_b['effect_size'].get('risk_diff_pp')}pp perm-p={st_b['perm_p_clustered']}")

    # ---- pooled ----
    print("\n" + "=" * 96)
    print("POOLED (cluster-robust): per-episode goal AND per-step progress")
    print("=" * 96)
    pools = {
        "self_adv_full": ("self-advertising full-goal boxes", self_adv_full),
        "all_full": ("ALL full-goal boxes (auth-milestone excluded)", full),
    }
    for key, (title, rows) in pools.items():
        eb = per_episode_block(rows, title)
        sb = per_step_block(rows, title)
        out["pooled"][key] = {"per_episode": eb, "per_step": sb}
        print(f"\n-- {title} --")
        print(fmt_block(eb))
        print(fmt_block(sb))

    # ---- stratified by vuln class (per-step progress, the powered metric) ----
    print("=" * 96)
    print("STRATIFIED by vuln class (per-step progress)")
    print("=" * 96)
    for vclass in ["RCE", "SQLi", "LFI", "auth"]:
        rows = [e for e in eps if e["vclass"] == vclass]
        if not rows:
            continue
        sb = per_step_block(rows, f"class={vclass}")
        out["stratified"][vclass] = sb
        print(fmt_block(sb))

    # ---- goal-aligned / weighted / milestone progress variants (full-goal boxes) ----
    print("=" * 96)
    print("PROGRESS VARIANTS (full-goal boxes) -- stricter than 'any abstract state change'")
    print("=" * 96)
    ga = per_step_block(full, "goal_aligned (forward-action progress)", "ga_steps", "goal_aligned_progress")
    out["progress_variants"]["goal_aligned"] = ga
    print(fmt_block(ga))
    # milestone (mean max-level reached) and weighted (means, with a permutation on the per-episode scalar)
    for arm in ("prm", "llm_only"):
        ms = [e["milestone"] for e in full if e["arm"] == arm]
        wp = [e["weighted_progress"] for e in full if e["arm"] == arm]
        out["progress_variants"].setdefault("milestone_mean", {})[arm] = round(float(np.mean(ms)), 3)
        out["progress_variants"].setdefault("weighted_progress_mean", {})[arm] = round(float(np.mean(wp)), 3)
    mm = out["progress_variants"]["milestone_mean"]
    wm = out["progress_variants"]["weighted_progress_mean"]
    print(f"  milestone_progress (mean max-level /5):  PRM {mm['prm']}  vs  llm_only {mm['llm_only']}")
    print(f"  weighted_progress_score (mean):          PRM {wm['prm']}  vs  llm_only {wm['llm_only']}")

    # ---- failure taxonomy ----
    print("\n" + "=" * 96)
    print("FAILURE TAXONOMY (per-episode terminal reason, by arm)")
    print("=" * 96)
    cats = ["success", "foothold_no_file", "exploit_executed_no_foothold", "exploit_never_proposed",
            "no_progress_exhausted", "budget_exhausted", "goal_unreachable_by_design", "safety_refusal"]
    tax = {arm: {c: 0 for c in cats} for arm in ("prm", "llm_only")}
    for e in eps:
        tax[e["arm"]][e["failure"]] += 1
    out["failure_taxonomy"] = tax
    print(f"  {'category':32s} {'PRM':>6s} {'llm_only':>10s}")
    for c in cats:
        print(f"  {c:32s} {tax['prm'][c]:>6d} {tax['llm_only'][c]:>10d}")

    # ---- multiple-comparison correction (Holm-Bonferroni) ----
    # PRIMARY confirmatory family = the 2 PRE-SPECIFIED pooled tests (per-step + per-episode over all
    # full-goal boxes). EXPLORATORY family = every clustered test (per-box + stratified + variants), which
    # is underpowered per-box by design. Report both so the headline is not penalised by exploratory tests.
    fam = _collect_pvalues(out)                              # full exploratory family
    holm = holm_bonferroni(fam)
    primary = [(n, p) for (n, p) in fam if n.startswith("pooled[all_full]")]
    holm_primary = holm_bonferroni(primary)
    n_sig = sum(1 for r in holm if r["significant"]); n_sug = sum(1 for r in holm if r["suggestive"])
    out["multiple_comparison"] = {
        "method": "holm-bonferroni", "alpha": 0.05,
        "primary_family": {"size": len(primary), "tests": holm_primary,
                           "note": "pre-specified confirmatory: pooled per-step + per-episode (all full-goal boxes)"},
        "exploratory_family": {"size": len(fam), "n_significant_after": n_sig, "n_suggestive": n_sug,
                               "results": holm}}
    print("\n" + "=" * 96)
    print("MULTIPLE-COMPARISON CORRECTION (Holm-Bonferroni)")
    print("=" * 96)
    print(f"  PRIMARY confirmatory family (n={len(primary)}, pre-specified pooled tests):")
    for r in holm_primary:
        v = "SIGNIFICANT" if r["significant"] else ("suggestive" if r["suggestive"] else "ns")
        print(f"    {r['test']:46s} raw={r['raw_p']:.4f}  adj={r['adj_p_holm']:.4f}  {v}")
    print(f"\n  EXPLORATORY family (n={len(fam)} all clustered tests): {n_sig} significant after Holm, "
          f"{n_sug} suggestive. Top:")
    for r in sorted(holm, key=lambda x: x["raw_p"])[:6]:
        v = "SIGNIFICANT" if r["significant"] else ("suggestive" if r["suggestive"] else "ns")
        print(f"    {r['test']:46s} raw={r['raw_p']:.4f}  adj={r['adj_p_holm']:.4f}  {v}")
    print("  (per-box tests are exploratory/underpowered by design; the confirmatory claim is the pooled family.)")

    (ROOT / "outputs" / "stage2_stats_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> outputs/stage2_stats_analysis.json  (seed={SEED}, n_perm={N_PERM}, n_boot={N_BOOT})")


if __name__ == "__main__":
    main()
