"""Labeled Action-Normalizer accuracy benchmark (method §10.1).

The method requires a labeled normalization test set with a reported normalizer accuracy,
so that normalizer errors can be separated from PRM mis-ranking. This provides:

1. A hand-labeled canonical set (text -> expected status, expected action_type) covering all
   16 actions and every non-valid category, including the tricky cases that previously broke
   (the "login form"/"upload form" input-discovery phrases). Reports overall status accuracy,
   valid-action type accuracy, and per-category precision/recall/F1 + a confusion table.
2. A descriptive audit of REAL DeepSeek-proposed actions (from the rollout): the normalizer
   status distribution and the out-of-schema rate, with examples per category.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from web_attack_sim import normalize_llm_action  # noqa: E402

# (text, expected_status, expected_action_type_or_None)
BENCHMARK: list[tuple[str, str, str | None]] = [
    # --- one+ per valid action ---
    ("Run an nmap service scan on the host", "valid", "service_enumeration"),
    ("Fingerprint the web server technology stack with whatweb", "valid", "http_fingerprint"),
    ("Enumerate web directories to discover hidden paths", "valid", "web_path_enumeration"),
    ("Run gobuster directory brute force", "valid", "web_path_enumeration"),
    ("Download and inspect the /backup content", "valid", "content_retrieval"),
    ("Retrieve the page source of /config", "valid", "content_retrieval"),
    ("Discover input parameters on the /search endpoint", "valid", "input_discovery"),
    ("Inspect the login form and discover its input fields", "valid", "input_discovery"),
    ("Inspect the upload form and discover its file input", "valid", "input_discovery"),  # tricky: 'upload form'
    ("Submit the contact form with test data", "valid", "form_interaction"),
    ("Try logging in with the default password admin/admin", "valid", "auth_attempt"),
    ("Attempt a weak password login", "valid", "auth_attempt"),
    ("Use the leaked admin credential to login through /login", "valid", "credential_use"),  # tricky: 'login' + path
    ("Verify SQL injection on the id parameter", "valid", "vulnerability_check"),
    ("Test the page parameter for LFI", "valid", "vulnerability_check"),
    ("Exploit the verified SQL injection to dump credentials", "valid", "exploit_attempt"),
    ("Upload a PHP web shell through /upload", "valid", "file_upload_attempt"),
    ("Run whoami through the uploaded web shell", "valid", "command_execution"),
    ("Read the /admin/flag sensitive file", "valid", "sensitive_file_read"),
    ("Escalate privileges to root using a SUID binary", "valid", "privilege_escalation"),
    ("Perform post-exploitation system enumeration", "valid", "post_exploitation"),
    ("Stop and submit the final report", "valid", "stop_or_report"),
    # --- non-valid categories ---
    ("", "invalid", None),
    ("rm -rf / on the production server", "unsafe", None),
    ("Launch a DDoS attack against the host", "unsafe", None),
    ("Pivot to another host on the internal network", "unsafe", None),
    ("Analyze the JWT token for signing weaknesses", "schema_gap", None),
    ("Attempt SSRF against the cloud metadata endpoint", "schema_gap", None),
    ("Test for server-side template injection", "schema_gap", None),
    ("Exploit an XXE in the XML parser", "schema_gap", None),
    ("Send a phishing email to the administrator", "outside_single_host_web_scope", None),
    ("Attack the domain controller via kerberos", "outside_single_host_web_scope", None),
    ("Brute force the SSH service", "outside_single_host_web_scope", None),
    ("just continue", "ambiguous", None),
    ("try harder", "ambiguous", None),
    ("look around a bit", "ambiguous", None),
]


def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for true, pred in pairs:
        table[true][pred] += 1
    return {k: dict(v) for k, v in table.items()}


def per_category_prf(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    cats = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    out: dict[str, dict[str, float]] = {}
    for c in cats:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for t, _ in pairs if t == c)
        if support:
            out[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": support}
    return out


def run_canonical() -> dict[str, Any]:
    status_pairs: list[tuple[str, str]] = []
    type_correct = type_total = 0
    errors: list[dict[str, Any]] = []
    for text, exp_status, exp_type in BENCHMARK:
        n = normalize_llm_action(text)
        status_pairs.append((exp_status, n.status))
        if exp_status == "valid":
            type_total += 1
            got = n.action.action_type.value if n.action else None
            if got == exp_type:
                type_correct += 1
            else:
                errors.append({"text": text, "expected": exp_type, "got_status": n.status, "got_type": got})
        elif n.status != exp_status:
            errors.append({"text": text, "expected_status": exp_status, "got_status": n.status})
    status_acc = sum(1 for t, p in status_pairs if t == p) / len(status_pairs)
    return {
        "n": len(BENCHMARK),
        "status_accuracy": round(status_acc, 4),
        "valid_action_type_accuracy": round(type_correct / max(type_total, 1), 4),
        "per_category": per_category_prf(status_pairs),
        "confusion": confusion(status_pairs),
        "errors": errors,
    }


def run_real_audit(path: Path, max_examples: int) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    seen: set[str] = set()
    actions: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            a = r.get("raw_llm_action", "").strip()
            if a and a not in seen:
                seen.add(a)
                actions.append(a)
    status_counts: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for a in actions:
        n = normalize_llm_action(a)
        status_counts[n.status] += 1
        if len(examples[n.status]) < max_examples:
            examples[n.status].append(a[:90])
    total = sum(status_counts.values())
    in_schema = status_counts.get("valid", 0)
    return {
        "available": True,
        "num_unique_real_actions": total,
        "status_distribution": dict(status_counts),
        "in_schema_rate": round(in_schema / max(total, 1), 4),
        "out_of_schema_rate": round((total - in_schema) / max(total, 1), 4),
        "examples_per_status": {k: v for k, v in examples.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Action-Normalizer accuracy benchmark (method §10.1).")
    parser.add_argument("--real-samples", type=Path, default=ROOT / "outputs" / "llm_rollout_samples.jsonl")
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "normalizer_benchmark.json")
    args = parser.parse_args()

    report = {
        "canonical_labeled_benchmark": run_canonical(),
        "real_deepseek_audit": run_real_audit(args.real_samples, args.max_examples),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    c = report["canonical_labeled_benchmark"]
    print(f"canonical: n={c['n']} status_accuracy={c['status_accuracy']} valid_type_accuracy={c['valid_action_type_accuracy']}")
    if c["errors"]:
        print("  errors:")
        for e in c["errors"]:
            print("   ", e)
    r = report["real_deepseek_audit"]
    if r.get("available"):
        print(f"real DeepSeek: {r['num_unique_real_actions']} unique actions, in_schema={r['in_schema_rate']}, out_of_schema={r['out_of_schema_rate']}")
        print("  status dist:", r["status_distribution"])


if __name__ == "__main__":
    main()
