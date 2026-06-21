"""Evaluate the Stage-2 ψ coverage layer: baseline (Stage-1 keyword ψ) vs enhanced.

Two evaluation sets, kept strictly separate:
  * DEV  = stage2/psi_benchmark.jsonl  — independent hand-labeled intents the recovery vocabulary
           was tuned on.
  * TEST = stage2/walkthroughs/*.json  — the Phase-1 fixtures, HELD OUT (the recovery layer was
           never tuned on their exact strings). This is the credible generalization number.

For each set we report, on the in-abstraction intents, the fraction the normalizer maps to the
correct action (accuracy) and the false-reject rate; and on the out-of-abstraction intents, the
false-accept rate (an out intent wrongly mapped to a valid action) — which MUST stay ~0.

  python -m stage2.eval_psi --report-output outputs/stage2_psi_eval.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.fixtures import OUT_OF_ABSTRACTION, iter_fixtures  # noqa: E402
from stage2.psi import EnhancedNormalizer  # noqa: E402
from web_attack_sim import normalize_llm_action  # noqa: E402


def _baseline(text: str):
    return normalize_llm_action(text)


def _score(pairs: list[tuple[str, str]], normalize) -> dict:
    """pairs = (intent, gold_label) where gold_label is one of the 16 actions or 'out_of_abstraction'."""
    in_total = in_correct = in_false_reject = 0
    out_total = out_false_accept = 0
    wrong = []
    for intent, gold in pairs:
        r = normalize(intent)
        valid = r.status == "valid"
        pred = r.action.action_type.value if r.action else None
        if gold == OUT_OF_ABSTRACTION:
            out_total += 1
            if valid:
                out_false_accept += 1
                wrong.append((intent, gold, "valid:" + str(pred)))
        else:
            in_total += 1
            if not valid:
                in_false_reject += 1
                wrong.append((intent, gold, r.status))
            elif pred == gold:
                in_correct += 1
            else:
                wrong.append((intent, gold, "wrong:" + str(pred)))
    return {
        "in_abstraction": in_total,
        "accuracy": round(in_correct / max(in_total, 1), 4),
        "correct": in_correct,
        "false_reject": in_false_reject,
        "false_reject_rate": round(in_false_reject / max(in_total, 1), 4),
        "out_abstraction": out_total,
        "false_accept": out_false_accept,
        "false_accept_rate": round(out_false_accept / max(out_total, 1), 4),
        "_wrong": wrong,
    }


def _load_benchmark(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pairs.append((d["intent"], d["label"]))
    return pairs


def _load_fixture_pairs(walkthroughs: Path) -> list[tuple[str, str]]:
    pairs = []
    for _path, data in iter_fixtures(walkthroughs):
        for step in data["steps"]:
            pairs.append((step["actor_intent"], step["reference_abstract_action"]))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline vs enhanced ψ accuracy (dev benchmark + held-out fixtures).")
    p.add_argument("--benchmark", type=Path, default=ROOT / "stage2" / "psi_benchmark.jsonl")
    p.add_argument("--walkthroughs", type=Path, default=ROOT / "stage2" / "walkthroughs")
    p.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "stage2_psi_eval.json")
    p.add_argument("--show-wrong", action="store_true", help="Print the residual mistakes on the held-out fixtures.")
    args = p.parse_args()

    enh = EnhancedNormalizer()
    dev = _load_benchmark(args.benchmark)
    test = _load_fixture_pairs(args.walkthroughs)

    out = {
        "dev_benchmark": {
            "n": len(dev),
            "baseline": _score(dev, _baseline),
            "enhanced": _score(dev, enh.normalize),
        },
        "heldout_fixtures": {
            "n": len(test),
            "baseline": _score(test, _baseline),
            "enhanced": _score(test, enh.normalize),
        },
    }

    def line(name, s):
        return (f"  {name:9s} acc={s['accuracy']:.3f}  false_reject={s['false_reject_rate']:.3f}  "
                f"false_accept={s['false_accept_rate']:.3f}  (in={s['in_abstraction']}, out={s['out_abstraction']})")

    print("DEV benchmark (tuned-on):")
    print(line("baseline", out["dev_benchmark"]["baseline"]))
    print(line("enhanced", out["dev_benchmark"]["enhanced"]))
    print("HELD-OUT fixtures (generalization):")
    print(line("baseline", out["heldout_fixtures"]["baseline"]))
    print(line("enhanced", out["heldout_fixtures"]["enhanced"]))

    if args.show_wrong:
        print("\nresidual mistakes on held-out fixtures (enhanced):")
        for intent, gold, got in out["heldout_fixtures"]["enhanced"]["_wrong"]:
            print(f"  gold={gold:22s} got={got:28s} <- {intent[:60]}")

    # strip the verbose _wrong lists from the persisted report (keep it compact)
    for grp in out.values():
        for k in ("baseline", "enhanced"):
            grp[k].pop("_wrong", None)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.report_output}")


if __name__ == "__main__":
    main()
