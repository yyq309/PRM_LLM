"""Validate a single walkthrough fixture: schema + φ parse coverage + ψ mapping.

Used by fixture authors (and the Phase-1 authoring workflow's verify stage) to confirm a new
box fixture is schema-valid, that φ can actually parse its tool outputs (low 'unparsed_outputs'),
and to surface the per-box honest metrics. Exit code is non-zero on schema failure.

  python -m stage2.validate_fixture stage2/walkthroughs/raven-2.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.fixtures import FixtureError, load_walkthrough  # noqa: E402
from stage2.replay import replay_one  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Validate one Stage-2 walkthrough fixture.")
    p.add_argument("fixture", type=Path)
    args = p.parse_args()

    try:
        data = load_walkthrough(args.fixture)
    except FixtureError as e:
        print(f"SCHEMA INVALID: {e}")
        raise SystemExit(2)

    # φ/ψ pass with no PRM (rerank not needed for validation).
    box = replay_one(args.fixture, data, prm=None)
    print(json.dumps({
        "box": box["box"],
        "abstract_family": box["abstract_family"],
        "n_steps": box["n_steps"],
        "out_of_abstraction_rate": box["out_of_abstraction_rate"],
        "psi": box["psi"],
        "phi_field_recall": box["phi"]["field_recall"],
        "phi_unparsed_outputs": box["phi"]["unparsed_outputs"],
        "schema_gap_tokens": box["schema_gap_tokens"],
    }, ensure_ascii=False, indent=2))

    warnings = []
    if box["phi"]["unparsed_outputs"]:
        warnings.append(f"{len(box['phi']['unparsed_outputs'])} tool output(s) produced no abstract facts")
    if box["phi"]["field_recall"] < 0.7 and box["phi"]["asserts"] > 0:
        warnings.append(f"low φ field recall {box['phi']['field_recall']:.2f} (<0.70) — check reference_state_after vs tool_output")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    print("\nSCHEMA OK")


if __name__ == "__main__":
    main()
