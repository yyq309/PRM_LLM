"""Stage-2 closed-loop runner (OFFLINE replay drive).

Exercises the full inference loop end-to-end without touching a network:

    propose candidates -> ψ normalize -> PRM rerank -> η (tool) -> ReplayExecutor -> φ observe -> repeat

The `ReplayExecutor` returns the recorded fixture output for whichever η-tool the PRM-chosen
action maps to, so the loop walks the recorded successful chain. We report a **chain-adherence**
proxy: the fraction of loop iterations where the PRM ranked the recorded-correct next action #1
among the policy's candidate set at the φ-reconstructed state. This is an honest *offline* proxy
for closed-loop fidelity — NOT a live success rate (that is Phase 2/3, gated).

    python -m stage2.closed_loop --walkthroughs stage2/walkthroughs --report-output outputs/stage2_closed_loop.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from stage2.eta import ETA_TOOL, ReplayExecutor  # noqa: E402
from stage2.fixtures import OUT_OF_ABSTRACTION, iter_fixtures  # noqa: E402
from stage2.phi import Phi, TOOL_ALIASES  # noqa: E402
from demo_pipeline import context_from_observation  # noqa: E402
from evaluate_prm_policy import policy_candidate_actions, precondition_guard_allows  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402
from web_attack_sim import normalize_llm_action  # noqa: E402


def _prm_rank(prm, context, candidates):
    samples = [{"context": context, "raw_llm_action": c,
                "normalized_action": normalize_llm_action(c).to_dict(),
                "normalizer_confidence": normalize_llm_action(c).confidence} for c in candidates]
    X = prm["vectorizer"].transform([extract_features(s) for s in samples]).toarray()
    scores = np.clip(prm["score"].predict(X), 0.0, 1.0)
    order = sorted(range(len(candidates)), key=lambda j: float(scores[j]), reverse=True)
    return order, [float(s) for s in scores]


def run_loop(path: Path, data: dict, prm) -> dict:
    in_steps = [s for s in data["steps"] if s["reference_abstract_action"] != OUT_OF_ABSTRACTION]
    phi = Phi(remaining_budget=len(data["steps"]) + 2)
    executor = ReplayExecutor(data["steps"])
    box = data["box"]
    trace_hist: list[dict] = []

    # env-faithful availability: an abstract action is only selectable while its opportunity
    # still exists. Offline we approximate the env action-mask by the η-tools still present in the
    # remaining recorded steps (an exhausted action — e.g. enumeration with nothing left to find —
    # would return no_new_information and be masked by the deployed loop's repeated-failure guard).
    def _route(tool: str) -> str:
        return TOOL_ALIASES.get(str(tool).lower(), str(tool).lower())

    iters = 0
    adhere = 0
    rows = []
    for k, step in enumerate(in_steps):
        obs = phi.observation()
        context = context_from_observation(box, k, obs, trace_hist)
        recorded_intent = step["actor_intent"]
        ref_action = step["reference_abstract_action"]
        obs_dict = obs.to_dict()
        remaining_routes = {_route(s["tool"]) for s in in_steps[k:]}
        # candidate set: the recorded-correct next action + the policy's state-derived proposals
        cands_all = [recorded_intent] + [c for c in policy_candidate_actions(obs_dict, 0, True)
                                         if c != recorded_intent]

        def _available(c: str) -> bool:
            n = normalize_llm_action(c)
            if n.action is None:
                return False
            if not precondition_guard_allows(n.to_dict(), obs_dict, None):
                return False
            return _route(ETA_TOOL.get(n.action.action_type, "")) in remaining_routes

        # mirror the DEPLOYED policy: PRM reranks only within the guard-allowed, still-available set.
        cands = [c for c in cands_all if _available(c)]
        if not cands:
            cands = [recorded_intent]
        order, scores = _prm_rank(prm, context, cands)
        top = cands[order[0]]
        top_action = normalize_llm_action(top).action
        top_type = top_action.action_type.value if top_action else None
        hit = int(top_type == ref_action)
        adhere += hit
        iters += 1

        # η -> ReplayExecutor -> φ (drive the recorded chain forward regardless, to advance state)
        chosen_action = normalize_llm_action(recorded_intent).action
        if chosen_action is not None:
            res = executor.run(chosen_action)
            phi.ingest(res.tool, res.output, target=step.get("target"),
                       meta={k2: step[k2] for k2 in ("path", "file_id", "vuln_id", "role", "yields_shell", "rooted") if k2 in step})
        trace_hist.append({"action": {"action_type": ref_action, "target": None, "parameter": None},
                           "feedback": {"success": True, "progress_event": None, "error_type": None, "evidence": ""}})
        rows.append({"iter": k, "recorded_action": ref_action, "prm_top_action": top_type,
                     "prm_ranked_correct_first": bool(hit), "eta_tool": ETA_TOOL.get(chosen_action.action_type) if chosen_action else None})

    final = phi.observation().to_dict()
    return {
        "box": box,
        "abstract_family": data["abstract_family"],
        "in_abstraction_steps": len(in_steps),
        "loop_iterations": iters,
        "chain_adherence": round(adhere / max(iters, 1), 4),
        "reached_shell": final["shell_state"] != "none",
        "reached_root": final["privilege_level"] == "root",
        "read_any_file": bool(final["read_files"]),
        "final_observation": final,
        "iters": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-2 OFFLINE closed-loop replay (propose->ψ->PRM->η->φ).")
    p.add_argument("--walkthroughs", type=Path, default=ROOT / "stage2" / "walkthroughs")
    p.add_argument("--prm-model", type=Path, default=ROOT / "outputs" / "prm_strong.joblib")
    p.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "stage2_closed_loop.json")
    args = p.parse_args()

    prm = joblib.load(args.prm_model)
    if prm.get("kind") != "strong":
        raise SystemExit("closed loop needs the strong PRM (kind=strong)")

    fixtures = iter_fixtures(args.walkthroughs)
    per_box = [run_loop(path, data, prm) for path, data in fixtures]
    n = len(per_box)
    agg = {
        "n_boxes": n,
        "mean_chain_adherence": round(sum(b["chain_adherence"] for b in per_box) / max(n, 1), 4),
        "boxes_replay_reached_shell": sum(int(b["reached_shell"]) for b in per_box),
        "boxes_replay_reached_root": sum(int(b["reached_root"]) for b in per_box),
        "note": ("chain_adherence = fraction of loop iterations where the abstract-trained PRM ranked the "
                 "recorded-correct next action #1 among the policy candidate set at the φ-reconstructed state. "
                 "Offline proxy for closed-loop fidelity; reached_shell/root reflect the REPLAYED chain "
                 "(driven by recorded output), not autonomous live success (Phase 2/3, gated)."),
    }
    report = {"stage": "stage2_closed_loop_offline", "summary": agg, "per_box": per_box}
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    for b in per_box:
        print(f"  {b['box']:22s} adherence={b['chain_adherence']:.2f}  "
              f"shell={b['reached_shell']} root={b['reached_root']}  ({b['abstract_family']})")


if __name__ == "__main__":
    main()
