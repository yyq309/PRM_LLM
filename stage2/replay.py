"""Stage-2 Phase-1: offline φ/ψ replay over recorded walkthroughs.

THE DECISIVE, NO-EXECUTION EXPERIMENT (STAGE2_PLAN.md §1, §3). For each recorded box
walkthrough it:
  1. replays the real tool output through φ to reconstruct the abstract `Observation`,
  2. normalizes each real operator intent through ψ (the Stage-1 normalizer),
  3. scores `candidate_pool` decision points with the abstract-trained strong PRM,
and measures, against the per-step HAND-LABELS:

  * out_of_abstraction_rate  — fraction of real steps that have NO abstract action (the
    make-or-break abstraction-gap number). Ground truth, not ψ-dependent.
  * psi_accuracy / psi_false_reject — does ψ map the mappable steps to the right action.
  * phi_field_recall          — does φ reconstruct the abstract-state fields each step set.
  * prm_rerank top-1 / pairwise — does the abstract-trained PRM prefer the good real action.

Decision gate: if out_of_abstraction_rate > 0.60, emit a prioritized schema-extension
shortlist (the aggregated `suggested_schema_extension` tokens) BEFORE building η/Phase-2.

NOTHING here contacts a network. Run:
  python -m stage2.replay --walkthroughs stage2/walkthroughs --report-output outputs/stage2_phase1_report.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib  # noqa: E402

from stage2.fixtures import OUT_OF_ABSTRACTION, iter_fixtures  # noqa: E402
from stage2.phi import Phi  # noqa: E402
from demo_pipeline import context_from_observation  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402
from web_attack_sim import normalize_llm_action  # noqa: E402

DEFAULT_GATE = 0.60


def _load_prm(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def _score_candidates(prm, context: str, candidates: list[str], normalize=normalize_llm_action) -> list[float] | None:
    """Strong-PRM score for each candidate — identical feature path to evaluate_prm_policy."""
    if prm is None or prm.get("kind") != "strong":
        return None
    samples = []
    for cand in candidates:
        n = normalize(cand)
        samples.append({
            "context": context,
            "raw_llm_action": cand,
            "normalized_action": n.to_dict(),
            "normalizer_confidence": n.confidence,
        })
    import numpy as np
    X = prm["vectorizer"].transform([extract_features(s) for s in samples]).toarray()
    return [float(v) for v in np.clip(prm["score"].predict(X), 0.0, 1.0)]


def _phi_meta(step: dict) -> dict:
    """Pass-through hints a fixture may give φ (path for forms, file_id, vuln_id, role...)."""
    return {k: step[k] for k in ("path", "file_id", "vuln_id", "role", "yields_shell", "rooted")
            if k in step}


def replay_one(path: Path, data: dict, prm, psi_cover=normalize_llm_action,
               psi_prm=normalize_llm_action) -> dict:
    """psi_cover normalizes the operator intent for the abstraction-coverage metric (and what η would
    execute). psi_prm normalizes the PRM candidate pool — it must stay the TRAINING-TIME normalizer,
    because the frozen strong PRM was trained on Stage-1-ψ features and goes out-of-distribution if
    fed a different normalizer's output."""
    phi = Phi(remaining_budget=len(data["steps"]) + 2)
    box = data["box"]
    steps_out = []
    trace_hist: list[dict] = []

    n_steps = len(data["steps"])
    n_out = 0
    psi_in_total = psi_correct = psi_false_reject = psi_wrong = 0
    psi_false_accept = 0  # out-of-abstraction step that ψ wrongly mapped to a valid action
    phi_assert_total = phi_assert_hit = 0
    phi_field_hits: Counter = Counter()
    phi_field_total: Counter = Counter()
    rerank_steps = rerank_top1 = 0
    pair_total = pair_hit = 0
    schema_gap_tokens: Counter = Counter()

    for i, step in enumerate(data["steps"]):
        ref = step["reference_abstract_action"]
        intent = step["actor_intent"]
        is_out = ref == OUT_OF_ABSTRACTION

        # --- decision point: PRM rerank on this state (state BEFORE this step's output) ---
        obs_before = phi.observation()
        context = context_from_observation(box, i, obs_before, trace_hist)
        rerank_info = None
        pool = step.get("candidate_pool")
        if pool and not is_out:
            scores = _score_candidates(prm, context, pool, psi_prm)
            if scores is not None:
                best = max(range(len(pool)), key=lambda j: scores[j])
                top1 = int(best == 0)
                rerank_steps += 1
                rerank_top1 += top1
                good = scores[0]
                wins = sum(1 for j in range(1, len(pool)) if good >= scores[j])
                pair_total += len(pool) - 1
                pair_hit += wins
                rerank_info = {"scores": [round(s, 4) for s in scores], "good_ranked_first": bool(top1)}

        # --- ψ: normalize the real operator intent (coverage metric) ---
        normalized = psi_cover(intent)
        nstatus = normalized.status
        naction = normalized.action.action_type.value if normalized.action else None
        if is_out:
            n_out += 1
            if step.get("suggested_schema_extension"):
                schema_gap_tokens[step["suggested_schema_extension"]] += 1
            if nstatus == "valid":
                psi_false_accept += 1
        else:
            psi_in_total += 1
            if nstatus != "valid":
                psi_false_reject += 1
            elif naction == ref:
                psi_correct += 1
            else:
                psi_wrong += 1

        # --- φ: ingest real tool output, then score field recall vs hand-label ---
        extracted = phi.ingest(step["tool"], step["tool_output"], target=step.get("target"), meta=_phi_meta(step))
        snap = phi.state.snapshot()
        rsa = step.get("reference_state_after", {})
        step_assert = step_hit = 0
        for fld, expected in rsa.items():
            if fld in {"auth_state", "shell_state", "privilege_level"}:
                step_assert += 1
                phi_field_total[fld] += 1
                ok = snap.get(fld) == expected
                step_hit += int(ok)
                phi_field_hits[fld] += int(ok)
            else:
                got = set(snap.get(fld, []))
                for val in expected:
                    step_assert += 1
                    phi_field_total[fld] += 1
                    ok = val in got
                    step_hit += int(ok)
                    phi_field_hits[fld] += int(ok)
        phi_assert_total += step_assert
        phi_assert_hit += step_hit

        # synthesize a trace entry for the next step's verbalized context
        trace_hist.append({
            "action": {"action_type": naction or "out_of_abstraction", "target": None, "parameter": None},
            "feedback": {"success": True, "progress_event": None,
                         "error_type": None if not is_out else "schema_gap",
                         "evidence": step.get("out_of_abstraction_reason") or ""},
        })

        steps_out.append({
            "step": i,
            "phase": step.get("phase"),
            "reference_abstract_action": ref,
            "psi_status": nstatus,
            "psi_action": naction,
            "psi_correct": (not is_out) and nstatus == "valid" and naction == ref,
            "phi_extracted": extracted,
            "phi_field_assert": step_assert,
            "phi_field_hit": step_hit,
            "rerank": rerank_info,
            "out_of_abstraction_reason": step.get("out_of_abstraction_reason"),
            "suggested_schema_extension": step.get("suggested_schema_extension"),
        })

    return {
        "box": box,
        "abstract_family": data["abstract_family"],
        "source": data["source"],
        "fixture": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        "n_steps": n_steps,
        "n_out_of_abstraction": n_out,
        "out_of_abstraction_rate": round(n_out / max(n_steps, 1), 4),
        "psi": {
            "in_abstraction_steps": psi_in_total,
            "accuracy": round(psi_correct / max(psi_in_total, 1), 4),
            "correct": psi_correct,
            "wrong_action": psi_wrong,
            "false_reject": psi_false_reject,
            "false_accept_on_out_steps": psi_false_accept,
        },
        "phi": {
            "field_recall": round(phi_assert_hit / max(phi_assert_total, 1), 4),
            "asserts": phi_assert_total,
            "hits": phi_assert_hit,
            "per_field_recall": {f: round(phi_field_hits[f] / phi_field_total[f], 4)
                                 for f in sorted(phi_field_total)},
            "unparsed_outputs": list(phi.state.unparsed_outputs),
        },
        "prm_rerank": {
            "decision_points": rerank_steps,
            "top1_rate": round(rerank_top1 / max(rerank_steps, 1), 4) if rerank_steps else None,
            "pairwise_accuracy": round(pair_hit / max(pair_total, 1), 4) if pair_total else None,
        },
        "schema_gap_tokens": dict(schema_gap_tokens),
        "steps": steps_out,
        "final_observation": phi.observation().to_dict(),
    }


def aggregate(per_box: list[dict], gate: float) -> dict:
    total_steps = sum(b["n_steps"] for b in per_box)
    total_out = sum(b["n_out_of_abstraction"] for b in per_box)
    psi_in = sum(b["psi"]["in_abstraction_steps"] for b in per_box)
    psi_corr = sum(b["psi"]["correct"] for b in per_box)
    psi_fr = sum(b["psi"]["false_reject"] for b in per_box)
    psi_fa = sum(b["psi"]["false_accept_on_out_steps"] for b in per_box)
    phi_assert = sum(b["phi"]["asserts"] for b in per_box)
    phi_hit = sum(b["phi"]["hits"] for b in per_box)
    rr_dp = sum(b["prm_rerank"]["decision_points"] for b in per_box)
    rr_t1 = sum(round(b["prm_rerank"]["top1_rate"] * b["prm_rerank"]["decision_points"])
                for b in per_box if b["prm_rerank"]["top1_rate"] is not None)

    tokens: Counter = Counter()
    for b in per_box:
        tokens.update(b["schema_gap_tokens"])

    ooa_rate = total_out / max(total_steps, 1)
    gate_tripped = ooa_rate > gate
    return {
        "n_boxes": len(per_box),
        "total_steps": total_steps,
        "out_of_abstraction_rate": round(ooa_rate, 4),
        "psi_accuracy_in_abstraction": round(psi_corr / max(psi_in, 1), 4),
        "psi_false_reject_rate": round(psi_fr / max(psi_in, 1), 4),
        "psi_false_accept_on_out_steps": psi_fa,
        "phi_field_recall": round(phi_hit / max(phi_assert, 1), 4),
        "prm_rerank_top1_rate": round(rr_t1 / max(rr_dp, 1), 4) if rr_dp else None,
        "prm_rerank_decision_points": rr_dp,
        "decision_gate": {
            "threshold": gate,
            "tripped": gate_tripped,
            "verdict": ("SCHEMA EXTENSION RECOMMENDED before Phase 2 — out-of-abstraction rate "
                        f"{ooa_rate:.2%} exceeds the {gate:.0%} gate."
                        if gate_tripped else
                        f"Abstraction covers the bulk of the real chains (out-of-abstraction "
                        f"{ooa_rate:.2%} <= {gate:.0%}); proceed to Phase 2 (η + single-box loop)."),
            "schema_extension_shortlist": [k for k, _ in tokens.most_common()],
            "schema_gap_token_counts": dict(tokens.most_common()),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-2 Phase-1 offline φ/ψ replay + abstraction-gap measurement.")
    p.add_argument("--walkthroughs", type=Path, default=ROOT / "stage2" / "walkthroughs")
    p.add_argument("--prm-model", type=Path, default=ROOT / "outputs" / "prm_strong.joblib")
    p.add_argument("--gate", type=float, default=DEFAULT_GATE)
    p.add_argument("--enhanced-psi", action="store_true",
                   help="Use the Stage-2 ψ coverage layer (stage2.psi.EnhancedNormalizer) for the "
                        "abstraction-coverage metric instead of the frozen Stage-1 keyword normalizer. "
                        "PRM candidate features stay on the training-time Stage-1 ψ (the frozen PRM is "
                        "out-of-distribution otherwise).")
    p.add_argument("--prm-uses-enhanced-psi", action="store_true",
                   help="DIAGNOSTIC: also feed the PRM candidate pool through the enhanced ψ. Demonstrates "
                        "the train/inference mismatch — rerank degrades because the frozen PRM was trained "
                        "on Stage-1-ψ features. Not the recommended Stage-2 configuration.")
    p.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "stage2_phase1_report.json")
    args = p.parse_args()

    fixtures = iter_fixtures(args.walkthroughs)
    if not fixtures:
        raise SystemExit(f"no walkthrough fixtures found in {args.walkthroughs}")
    prm = _load_prm(args.prm_model)

    if args.enhanced_psi:
        from stage2.psi import EnhancedNormalizer
        psi_cover = EnhancedNormalizer().normalize
    else:
        psi_cover = normalize_llm_action
    # The frozen strong PRM was trained on Stage-1-ψ features; keep its candidate normalization there
    # unless the diagnostic flag explicitly asks to show the mismatch.
    psi_prm = psi_cover if args.prm_uses_enhanced_psi else normalize_llm_action

    per_box = [replay_one(path, data, prm, psi_cover, psi_prm) for path, data in fixtures]
    summary = aggregate(per_box, args.gate)
    report = {
        "stage": "stage2_phase1_offline_replay",
        "psi_variant": "enhanced" if args.enhanced_psi else "stage1_baseline",
        "prm_candidate_psi": "enhanced" if args.prm_uses_enhanced_psi else "stage1_training_time",
        "prm_model": str(args.prm_model),
        "prm_loaded": prm is not None,
        "summary": summary,
        "per_box": per_box,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nper-box out-of-abstraction / ψ-acc / φ-recall / PRM-top1:")
    for b in per_box:
        rr = b["prm_rerank"]["top1_rate"]
        print(f"  {b['box']:22s} ooa={b['out_of_abstraction_rate']:.2f}  "
              f"ψ={b['psi']['accuracy']:.2f}  φ={b['phi']['field_recall']:.2f}  "
              f"PRM-top1={rr if rr is None else round(rr,2)}  ({b['abstract_family']})")
    print(f"\nreport -> {args.report_output}")


if __name__ == "__main__":
    main()
