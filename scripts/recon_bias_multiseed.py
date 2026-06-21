"""#3: multi-seed recon-bias variance (quantifies the H finding that the recon over-valuation is
SEED-DEPENDENT, not a robust/deterministic property).

For each oracle seed's PRM dataset, compute the recon over-valuation signal — web_path_enumeration's
mean PRM target label OVERALL and in the ADVANCED (post-foothold) phase, plus the recon-class vs
exploit-class label in advanced states. Report per-seed + mean/std/range across seeds, and compare to
the DEPLOYED (seed-gated) PRM dataset. If the across-seed range is wide and the deployed value sits at
the high end, the deployed PRM's strong recon bias is a seed-GATE selection artifact (H confirmed).

    python scripts/recon_bias_multiseed.py
-> outputs/recon_bias_multiseed.json (+ table)
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "recon_bias_multiseed.json"

RECON = {"http_fingerprint", "web_path_enumeration", "content_retrieval", "service_enumeration",
         "input_discovery"}
EXPLOIT = {"exploit_attempt", "command_execution", "privilege_escalation", "file_upload_attempt",
           "sensitive_file_read"}


def _advanced(context: str) -> bool:
    m = re.search(r"Shell state:\s*(\w+)", context)
    return (m.group(1) if m else "none") in ("webshell", "command_execution")


def metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    wpe_all, wpe_adv, recon_adv, exploit_adv, cmd_adv = [], [], [], [], []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        na = r["normalized_action"]
        at = (na if isinstance(na, dict) else {}).get("action_type")
        if not at:
            continue
        s = float(r["score"])
        adv = _advanced(r["context"])
        if at == "web_path_enumeration":
            wpe_all.append(s)
            if adv:
                wpe_adv.append(s)
        if at == "command_execution" and adv:
            cmd_adv.append(s)
        if adv and at in RECON:
            recon_adv.append(s)
        if adv and at in EXPLOIT:
            exploit_adv.append(s)
    m = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {"web_path_enum_overall": m(wpe_all), "web_path_enum_advanced": m(wpe_adv),
            "command_exec_advanced": m(cmd_adv), "recon_advanced": m(recon_adv),
            "exploit_advanced": m(exploit_adv), "n_wpe_adv": len(wpe_adv)}


def main() -> None:
    seeds = {f"seed{s}": ROOT / "outputs" / f"prm_samples_seed{s}_train.jsonl" for s in range(5)}
    refs = {"DEPLOYED (seed-gated)": ROOT / "outputs" / "prm_samples_train.jsonl",
            "reward-fixed seed0": ROOT / "outputs" / "prm_samples_rewardfix_train.jsonl"}

    per_seed = {k: metrics(p) for k, p in seeds.items()}
    per_seed = {k: v for k, v in per_seed.items() if v}
    ref_metrics = {k: metrics(p) for k, p in refs.items()}

    def across(key):
        vals = [v[key] for v in per_seed.values() if v.get(key) is not None]
        return {"mean": round(st.mean(vals), 3), "std": round(st.pstdev(vals), 3),
                "min": min(vals), "max": max(vals), "n_seeds": len(vals)} if vals else None

    summary = {k: across(k) for k in ("web_path_enum_overall", "web_path_enum_advanced", "command_exec_advanced")}
    out = {"per_seed": per_seed, "across_seeds": summary, "reference": ref_metrics}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== recon over-valuation per seed (web_path_enum label) ===")
    print(f"{'seed':22s} {'wpe_overall':>12s} {'wpe_advanced':>13s} {'cmd_exec_adv':>13s} {'n_adv':>6s}")
    for k, v in per_seed.items():
        print(f"{k:22s} {str(v['web_path_enum_overall']):>12s} {str(v['web_path_enum_advanced']):>13s} "
              f"{str(v['command_exec_advanced']):>13s} {v['n_wpe_adv']:>6d}")
    print("  " + "-" * 60)
    for k, v in ref_metrics.items():
        if v:
            print(f"{k:22s} {str(v['web_path_enum_overall']):>12s} {str(v['web_path_enum_advanced']):>13s} "
                  f"{str(v['command_exec_advanced']):>13s} {v['n_wpe_adv']:>6d}")
    print("\n=== across-seed variance (the H 'seed-dependent' claim, quantified) ===")
    for k, v in summary.items():
        if v:
            print(f"  {k:24s} mean={v['mean']} std={v['std']} range=[{v['min']}, {v['max']}] (n={v['n_seeds']} seeds)")
    dep = ref_metrics.get("DEPLOYED (seed-gated)")
    adv = summary.get("web_path_enum_advanced")
    if dep and adv and dep["web_path_enum_advanced"] is not None:
        hi = dep["web_path_enum_advanced"] >= adv["max"]
        print(f"\nDEPLOYED web_path_enum_advanced={dep['web_path_enum_advanced']} vs seed range "
              f"[{adv['min']},{adv['max']}] -> deployed at/above the {'HIGH end (selection artifact confirmed)' if hi else 'within range'}")
    print(f"\nreport -> {OUT}")


if __name__ == "__main__":
    main()
