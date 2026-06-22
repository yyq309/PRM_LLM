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

print("wrote 5 figures to", os.path.abspath(OUT))
for f in sorted(os.listdir(OUT)):
    print("  ", f)
