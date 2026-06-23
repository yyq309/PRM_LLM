"""Generate publication figures for the Experiments chapter from the real result JSONs.

All numbers are the verified values reported in STAGE2_LIVE_RESULTS.md / the eval reports. Run:
    python scripts/make_paper_figures.py
Outputs PNGs (300 dpi) to figures/.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

PRM_C, LLM_C = "#2a6f97", "#e6a23c"   # advisor vs raw-LLM
CHANCE = "#9aa0a6"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 300, "savefig.bbox": "tight"})


def _bars(ax, groups, prm, llm, ylabel, title, pct=True, annot=True):
    x = np.arange(len(groups)); w = 0.38
    b1 = ax.bar(x - w/2, prm, w, label="prm (advisor re-ranks)", color=PRM_C)
    b2 = ax.bar(x + w/2, llm, w, label="llm_only (raw LLM order)", color=LLM_C)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, weight="bold")
    if pct:
        ax.set_ylim(0, 109)
        ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")
    if annot:
        for bs in (b1, b2):
            for b in bs:
                h = b.get_height()
                ax.text(b.get_x()+b.get_width()/2, h+1.5, f"{h:.0f}" + ("%" if pct else ""),
                        ha="center", va="bottom", fontsize=8.5)
    return b1, b2


# ---- Fig 1: multi-LLM proposer-conditional (the money figure) ----
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
vend = ["DeepSeek", "Qwen-3.7", "GPT-5.4"]
_bars(a, vend, [100, 100, 100], [56, 100, 100], "root rate",
      "DC-1 full chain  (web -> foothold -> root)")
a.annotate("PRM rescues the\nweak proposer", xy=(0, 56), xytext=(0.05, 30),
           fontsize=8.5, ha="left", arrowprops=dict(arrowstyle="->", color="#444"))
a.annotate("strong proposers\nalready saturate", xy=(1.4, 100), xytext=(1.1, 70), fontsize=8.5, ha="left")
_bars(b, vend, [100, 100, 60], [40, 40, 0], "goal rate",
      "Joomla CVE-2017-8917  (3-vendor rescue)")
b.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
fig.suptitle("Figure 1.  The advisor rescues a struggling proposer; it is redundant for a strong one "
             "(n=18 DC-1 deepseek pooled, n=5 web)", fontsize=10.5, y=1.02)
fig.savefig(os.path.join(OUT, "fig1_multillm_proposer_conditional.png")); plt.close(fig)

# ---- Fig 2: top-1 ranking accuracy, 3 vendors x 7 boxes (20/21) ----
boxes = ["dc1", "joomla", "php-cgi", "struts2-045", "struts2-048", "thinkphp-5", "thinkphp-5023"]
data = {
    "DeepSeek": ([.580, .264, .550, .333, .333, .447, .480], [.361, .403, .229, .000, .100, .317, .317]),
    "Qwen-3.7": ([.474, .321, .783, .450, .767, .767, .633], [.409, .249, .142, .000, .000, .317, .317]),
    "GPT-5.4":  ([.604, .286, .600, .767, .517, .700, .600], [.317, .260, .189, .000, .000, .173, .220]),
}
fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
for ax, (v, (prm, llm)) in zip(axes, data.items()):
    x = np.arange(len(boxes)); w = 0.38
    ax.bar(x - w/2, prm, w, color=PRM_C, label="prm")
    ax.bar(x + w/2, llm, w, color=LLM_C, label="llm_only")
    ax.axhline(0.5, ls="--", lw=1, color=CHANCE)
    ax.set_ylim(0, 1.0); ax.set_ylabel(f"{v}\ntop-1 acc"); ax.set_xticks(x); ax.set_xticklabels(boxes)
    if v == "DeepSeek":
        ax.annotate("only exception\n(prm<llm) — but prm\nstill wins GOAL here",
                    xy=(1, .264), xytext=(1.3, .62), fontsize=8, color="#b00",
                    arrowprops=dict(arrowstyle="->", color="#b00"))
        ax.legend(loc="upper right", ncol=2, fontsize=8.5)
axes[0].set_title("Figure 2.  Top-1 ranking accuracy: prm > llm_only on 20 of 21 vendor-boxes "
                  "(dashed = 0.5 chance)", fontsize=10.5, weight="bold")
fig.savefig(os.path.join(OUT, "fig2_top1_ranking.png")); plt.close(fig)

# ---- Fig 3: DC-1 phase split (mechanism) ----
fig, ax = plt.subplots(figsize=(6.5, 4.3))
_bars(ax, ["Web phase", "Local / privesc phase"], [36.4, 36.7], [31.5, 9.3],
      "per-step progress rate", "Figure 3.  DC-1: where the advisor earns its keep", pct=True)
ax.axhline(9.3, xmin=0.5, xmax=0.97, ls=":", lw=1, color="#b00")
ax.annotate("raw LLM COLLAPSES in the\nprivilege-escalation phase (9%)", xy=(1.19, 9.3),
            xytext=(0.55, 55), fontsize=9, color="#b00",
            arrowprops=dict(arrowstyle="->", color="#b00"))
ax.legend(loc="upper left", fontsize=8.5)
fig.savefig(os.path.join(OUT, "fig3_dc1_phase_split.png")); plt.close(fig)

# ---- Fig 4: recon over-valuation (C-C mechanism) ----
acts = ["file_upload", "web_path_enum", "vuln_check", "exploit_attempt", "sensitive_file_read",
        "auth_attempt", "input_discovery", "http_fingerprint", "credential_use",
        "command_execution", "privilege_escalation", "content_retrieval"]
vals = [1.000, 0.887, 0.637, 0.535, 0.486, 0.344, 0.165, 0.126, 0.086, 0.044, 0.040, 0.021]
# recon/scouting (warm) vs decisive kill-chain action (cool)
recon = {"file_upload", "web_path_enum", "vuln_check", "input_discovery", "http_fingerprint"}
fig, ax = plt.subplots(figsize=(7.5, 5))
colors = ["#c1666b" if a in recon else "#4d908e" for a in acts]
y = np.arange(len(acts))[::-1]
ax.barh(y, vals, color=colors)
for yi, vv in zip(y, vals):
    ax.text(vv + 0.012, yi, f"{vv:.2f}", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(acts); ax.set_xlim(0, 1.12)
ax.set_xlabel("PRM mean label (higher = advisor thinks more valuable)")
ax.set_title("Figure 4.  The advisor over-values reconnaissance\n"
             "(scouting actions rated high; decisive late-chain actions rated low)",
             fontsize=10.5, weight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#c1666b", label="reconnaissance / scouting"),
                   Patch(color="#4d908e", label="exploit / foothold / privesc")],
          loc="lower right", fontsize=9)
fig.savefig(os.path.join(OUT, "fig4_recon_overvaluation.png")); plt.close(fig)

# ---- Fig 5: Stage-1 pairwise across splits (deployed PRM) ----
fig, ax = plt.subplots(figsize=(6.5, 4.3))
splits = ["all held-out", "new instances\n(seen chains)", "new chain\nshapes (hard)"]
pw = [0.89, 0.98, 0.80]
err = [[0.89-0.843], [0.937-0.89]]  # asym CI on the 'all' bar only
bars = ax.bar(splits, pw, color=[PRM_C, PRM_C, PRM_C], width=0.6)
ax.errorbar([0], [0.89], yerr=err, fmt="none", ecolor="#1b3b53", capsize=5, lw=1.5)
ax.axhline(0.5, ls="--", lw=1, color=CHANCE); ax.text(2.35, 0.515, "chance", color=CHANCE, fontsize=8.5)
ax.bar([2], [0.93], width=0.6, color="none", edgecolor="#2a6f97", ls="--", lw=1.5)
ax.annotate("0.93 (preference-loss\nvariant, not deployed)", xy=(2, 0.93), xytext=(0.9, 0.965),
            fontsize=8, arrowprops=dict(arrowstyle="->", color="#444"))
for i, vv in enumerate(pw):
    ax.text(i, vv-0.06, f"{vv:.2f}", ha="center", color="white", fontsize=10, weight="bold")
ax.set_ylim(0.4, 1.02); ax.set_ylabel("pairwise ranking accuracy")
ax.set_title("Figure 5.  Stage-1 advisor quality (deployed PRM, 5 seeds; CI on 'all')",
             fontsize=10.5, weight="bold")
fig.savefig(os.path.join(OUT, "fig5_stage1_pairwise.png")); plt.close(fig)

# ---- Fig 6: process / stage-level metrics (16 boxes pooled; all rendered higher-is-better) ----
# labels, prm, llm, significant? (clustered p<0.05)
metrics = [
    ("per-step\nprogress", 52.7, 37.6, True),
    ("goal-aligned\nprogress", 26.3, 15.0, True),
    ("milestone stage\n(/5 ×20)", 32.9, 21.2, True),
    ("weighted\nprogress (×100)", 59.2, 45.0, True),
    ("foothold/shell\nreach", 30.8, 20.0, False),
    ("step efficiency\n(1−wasted)", 65.0, 51.0, False),
]
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(metrics)); w = 0.38
prm_v = [m[1] for m in metrics]; llm_v = [m[2] for m in metrics]
b1 = ax.bar(x - w/2, prm_v, w, color=PRM_C, label="prm (advisor)")
b2 = ax.bar(x + w/2, llm_v, w, color=LLM_C, label="llm_only")
for i, m in enumerate(metrics):
    ax.text(i - w/2, m[1] + 1.5, f"{m[1]:.0f}", ha="center", fontsize=8.5)
    ax.text(i + w/2, m[2] + 1.5, f"{m[2]:.0f}", ha="center", fontsize=8.5)
    if m[3]:
        ax.text(i, max(m[1], m[2]) + 6, "★", ha="center", color="#b00", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=8.5)
ax.set_ylim(0, 80); ax.set_ylabel("score (higher = better)")
ax.set_title("Figure 6.  Process / stage-level metrics, prm vs llm_only (16 web boxes pooled). "
             "★ = clustered-significant (p<0.05)", fontsize=10, weight="bold")
ax.legend(loc="upper right", fontsize=9)
fig.savefig(os.path.join(OUT, "fig6_process_metrics.png")); plt.close(fig)

# ---- Fig 7: cross-VM phase split (the advisor's value lives in the privesc phase) ----
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (vm, web, loc) in zip((a, b), [
        ("DC-1", (36, 32), (37, 9)), ("Symfonos:1", (51, 48), (100, 24))]):
    xx = np.arange(2); ww = 0.38
    ax.bar(xx - ww/2, [web[0], loc[0]], ww, color=PRM_C, label="prm")
    ax.bar(xx + ww/2, [web[1], loc[1]], ww, color=LLM_C, label="llm_only")
    for j, (p, l) in enumerate([web, loc]):
        ax.text(j - ww/2, p + 1.5, f"{p}", ha="center", fontsize=9)
        ax.text(j + ww/2, l + 1.5, f"{l}", ha="center", fontsize=9)
    ax.set_xticks(xx); ax.set_xticklabels(["web phase", "privesc phase"])
    ax.set_ylim(0, 109); ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")
    ax.set_title(vm, fontsize=11, weight="bold")
    ax.annotate("raw LLM\ncollapses", xy=(1 + ww/2, loc[1]), xytext=(0.55, 60),
                fontsize=8.5, color="#b00", arrowprops=dict(arrowstyle="->", color="#b00"))
a.set_ylabel("per-step progress rate"); b.legend(loc="upper left", fontsize=9)
fig.suptitle("Figure 7.  Per-step progress by phase: the advisor's value is concentrated in the "
             "privilege-escalation phase, on both VMs", fontsize=10, y=1.03)
fig.savefig(os.path.join(OUT, "fig7_phase_split_2vm.png")); plt.close(fig)

# ---- Fig 8: reranker-isolation ablation (only the ranker is swapped; candidate set fixed) ----
GREY = "#9aa0a6"; ORC = "#e0a526"
depl = [("llm_only", 48.1, LLM_C), ("random", 50.0, GREY), ("oracle*", 47.9, ORC), ("prm (ours)", 68.5, PRM_C)]
strt = [("random", 29.3, GREY), ("oracle", 31.2, ORC), ("prm (ours)", 26.7, PRM_C)]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3))
for ax, data, ylim, title in [
        (a, depl, 80, "Deployed: real LLM proposer (n=30/arm, 6 boxes)"),
        (b, strt, 44, "Stress test: full action surface (n=80/arm, 11 boxes)")]:
    xs = np.arange(len(data))
    ax.bar(xs, [d[1] for d in data], 0.62, color=[d[2] for d in data],
           edgecolor=[(PRM_C if d[0].startswith("prm") else "none") for d in data], linewidth=2)
    for x, d in zip(xs, data):
        ax.text(x, d[1] + ylim*0.015, f"{d[1]:.1f}", ha="center", fontsize=9,
                weight=("bold" if d[0].startswith("prm") else "normal"))
    ax.set_xticks(xs); ax.set_xticklabels([d[0] for d in data], fontsize=9, rotation=10)
    ax.set_ylim(0, ylim); ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")
    ax.set_title(title, fontsize=10, weight="bold")
a.set_ylabel("per-step progress rate")
a.annotate("PRM beats raw LLM, random,\nand the value-oracle (all p<0.01)",
           xy=(3, 68.5), xytext=(-0.35, 72), fontsize=8.5, color=PRM_C,
           arrowprops=dict(arrowstyle="->", color=PRM_C))
b.annotate("fed the full menu, PRM's recon\nbias drops it below random",
           xy=(2, 27.2), xytext=(0.0, 40), fontsize=8.3, color="#b00",
           arrowprops=dict(arrowstyle="->", color="#b00"))
fig.suptitle("Figure 8.  Reranker-isolation: only the ranker is swapped (candidate set fixed). The PRM's "
             "value is proposer-conditional.  *oracle ranks the abstract optimum via ψ (not a clean UB here).",
             fontsize=9, y=1.02)
fig.savefig(os.path.join(OUT, "fig8_reranker_isolation.png")); plt.close(fig)

print("wrote 8 figures to", os.path.abspath(OUT))
for f in sorted(os.listdir(OUT)):
    print("  ", f)
