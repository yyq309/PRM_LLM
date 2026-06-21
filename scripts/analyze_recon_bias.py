"""G2(a): (action_type x phase) label histogram on the PRM training data — the C-C mechanism evidence.

The claim (C-C): the MASKED abstract oracle rarely produces "recon-when-already-advanced" states, so the
PRM over-values recon (web_path_enumeration) and extrapolates that to real targets. This script tests it
DIRECTLY on the training labels: split each candidate's PRM target `score` by the state PHASE (derived
from the context's `Shell state:` / `Verified vulnerabilities:`), and show whether recon stays high-valued
in ADVANCED states (post-foothold) — and crucially how FEW such samples exist (the distribution gap).

    python scripts/analyze_recon_bias.py
-> outputs/recon_bias_histogram.json  (+ printed table)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import argparse

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "outputs" / "prm_samples_train.jsonl"
OUT = ROOT / "outputs" / "recon_bias_histogram.json"

RECON = {"http_fingerprint", "web_path_enumeration", "content_retrieval", "service_enumeration",
         "input_discovery"}
EXPLOIT = {"exploit_attempt", "command_execution", "privilege_escalation", "file_upload_attempt",
           "sensitive_file_read"}


def _phase(context: str) -> str:
    """Derive the state phase from the context string (observable state only)."""
    shell = re.search(r"Shell state:\s*(\w+)", context)
    shell = shell.group(1) if shell else "none"
    vulns = re.search(r"Verified vulnerabilities:\s*\[([^\]]*)\]", context)
    has_vuln = bool(vulns and vulns.group(1).strip())
    if shell in ("webshell", "command_execution"):
        return "advanced_foothold"      # a shell is already established -> recon is redundant here
    if has_vuln:
        return "mid_vuln_found"          # a vuln is verified but no shell yet
    return "early_recon"                  # no foothold, no vuln -> recon is genuinely useful


def main() -> None:
    ap = argparse.ArgumentParser(description="(action x phase) PRM-label recon-bias histogram (G2).")
    ap.add_argument("--input", type=Path, default=SAMPLES)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    global OUT
    OUT = args.output
    recs = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    # cell[(action_type, phase)] -> list of labels
    cell: dict = defaultdict(list)
    by_action: dict = defaultdict(list)
    for r in recs:
        na = r["normalized_action"]
        at = (na if isinstance(na, dict) else {}).get("action_type")
        if not at:
            continue
        label = float(r["score"])
        ph = _phase(r["context"])
        cell[(at, ph)].append(label)
        by_action[at].append(label)

    phases = ["early_recon", "mid_vuln_found", "advanced_foothold"]
    actions = sorted(by_action, key=lambda a: -sum(by_action[a]) / len(by_action[a]))

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    table = {}
    for at in actions:
        row = {"overall_mean": mean(by_action[at]), "overall_n": len(by_action[at])}
        for ph in phases:
            xs = cell.get((at, ph), [])
            row[ph] = {"mean": mean(xs), "n": len(xs)}
        table[at] = row

    # the smoking gun: recon vs exploit label IN ADVANCED states + how few recon-in-advanced samples exist
    def cls_mean(cls, ph):
        xs = [x for a in cls for x in cell.get((a, ph), [])]
        return mean(xs), len(xs)
    recon_adv = cls_mean(RECON, "advanced_foothold")
    exploit_adv = cls_mean(EXPLOIT, "advanced_foothold")
    recon_early = cls_mean(RECON, "early_recon")
    summary = {
        "recon_label_advanced": {"mean": recon_adv[0], "n": recon_adv[1]},
        "exploit_label_advanced": {"mean": exploit_adv[0], "n": exploit_adv[1]},
        "recon_label_early": {"mean": recon_early[0], "n": recon_early[1]},
        "web_path_enum_advanced": {"mean": mean(cell.get(("web_path_enumeration", "advanced_foothold"), [])),
                                   "n": len(cell.get(("web_path_enumeration", "advanced_foothold"), []))},
        "web_path_enum_early": {"mean": mean(cell.get(("web_path_enumeration", "early_recon"), [])),
                                "n": len(cell.get(("web_path_enumeration", "early_recon"), []))},
    }
    OUT.write_text(json.dumps({"summary": summary, "table": table}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("=== (action_type x phase) mean PRM target label  [mean (n)] ===")
    print(f"{'action_type':24s} {'overall':>13s} {'early_recon':>16s} {'mid_vuln':>14s} {'advanced':>16s}")
    for at in actions:
        row = table[at]
        def c(ph):
            x = row[ph]; return f"{x['mean']} ({x['n']})" if x["mean"] is not None else "-"
        print(f"{at:24s} {str(row['overall_mean'])+' ('+str(row['overall_n'])+')':>13s} "
              f"{c('early_recon'):>16s} {c('mid_vuln_found'):>14s} {c('advanced_foothold'):>16s}")
    print("\n=== C-C smoking gun ===")
    print(f"  RECON label in ADVANCED (post-foothold) states: mean={summary['recon_label_advanced']['mean']} "
          f"n={summary['recon_label_advanced']['n']}")
    print(f"  EXPLOIT-class label in ADVANCED states:         mean={summary['exploit_label_advanced']['mean']} "
          f"n={summary['exploit_label_advanced']['n']}")
    print(f"  web_path_enum: early={summary['web_path_enum_early']['mean']} (n={summary['web_path_enum_early']['n']}) "
          f"vs advanced={summary['web_path_enum_advanced']['mean']} (n={summary['web_path_enum_advanced']['n']})")
    print(f"\nreport -> {OUT}")


if __name__ == "__main__":
    main()
