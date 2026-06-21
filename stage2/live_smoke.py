"""Stage-2 LIVE smoke test — validate the real execution pipeline on ONE authorized lab target.

This is the first live milestone: prove that η renders a real, box-specific command, the gated
`LiveExecutor` runs it against the real target, and φ reconstructs the abstract state from the REAL
response — end to end, safely. It runs a FIXED, read-only recon→RCE sequence (not the autonomous
PRM loop, which additionally needs the tools image + an LLM proposer). Commands are limited to
`whoami` / `id` / `cat /etc/passwd` through the target's `eta_recipes`.

Refuses to do anything unless the safety gate authorizes it:
    $env:STAGE2_LIVE_AUTHORIZED = "i-own-this-isolated-authorized-lab"
    python -m stage2.live_smoke --target stage2/targets/thinkphp-5-rce.json --confirmed-isolated
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.eta import (LiveExecutor, eta_ctx_from_target, eta_recipes_from_target, load_target)  # noqa: E402
from stage2.phi import Phi  # noqa: E402
from stage2.safety import AuditLog, AuthorizationGate  # noqa: E402
from web_attack_sim.action_space import ActionType  # noqa: E402
from web_attack_sim.schemas import Action  # noqa: E402

# Fixed, read-only sequence. (action, φ-meta) — the meta is the hand-known context η/φ would carry.
SEQUENCE = [
    (ActionType.HTTP_FINGERPRINT, {}),
    (ActionType.VULNERABILITY_CHECK, {"vuln_id": "thinkphp_rce"}),
    (ActionType.EXPLOIT_ATTEMPT, {"vuln_id": "thinkphp_rce", "yields_shell": True}),
    (ActionType.COMMAND_EXECUTION, {}),
    (ActionType.SENSITIVE_FILE_READ, {"file_id": "etc_passwd"}),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-2 LIVE smoke test (gated; fixed recon→RCE sequence).")
    p.add_argument("--target", type=Path, default=ROOT / "stage2" / "targets" / "thinkphp-5-rce.json")
    p.add_argument("--confirmed-isolated", action="store_true",
                   help="Operator asserts an owned, isolated, snapshot/restore lab. Required with the "
                        "STAGE2_LIVE_AUTHORIZED env confirmation string.")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "stage2_live_smoke.json")
    args = p.parse_args()

    desc = load_target(args.target)
    target = desc["target"]
    gate = AuthorizationGate(confirmed_isolated=args.confirmed_isolated)
    if not gate.authorized():
        raise SystemExit("LIVE smoke refused: " + gate.reason_blocked() +
                         "\nSet STAGE2_LIVE_AUTHORIZED and pass --confirmed-isolated only for an owned, "
                         "isolated, authorized lab.")

    audit = AuditLog(ROOT / "outputs" / "stage2_live_smoke_audit.jsonl")
    executor = LiveExecutor(target, gate=gate, audit=audit, timeout=args.timeout,
                            recipes=eta_recipes_from_target(desc), **eta_ctx_from_target(desc))
    phi = Phi(remaining_budget=len(SEQUENCE) + 2)

    print(f"LIVE smoke against {desc['name']} ({target})\n")
    steps = []
    for action_type, meta in SEQUENCE:
        action = Action(action_type=action_type)
        res = executor.run(action)
        extracted = phi.ingest(res.tool, res.output, target=target, meta=meta)
        snap = phi.state.snapshot()
        head = " ".join(res.output.split())[:120]
        print(f"[{action_type.value:20s}] tool={res.tool:8s} executed={res.executed}")
        print(f"    output: {head}")
        print(f"    φ extracted: {extracted}")
        steps.append({"action": action_type.value, "tool": res.tool, "executed": res.executed,
                      "output_head": head, "phi_extracted": extracted, "state": snap})

    final = phi.observation().to_dict()
    result = {
        "stage": "stage2_live_smoke", "target": desc["name"], "target_url": target,
        "reached_command_execution": final["shell_state"] in {"command_execution", "webshell"},
        "reached_root": final["privilege_level"] == "root",
        "read_files": final["read_files"],
        "final_observation": final, "steps": steps,
        "note": ("Fixed-sequence LIVE pipeline validation (η→gated LiveExecutor→real target→φ). NOT the "
                 "autonomous PRM loop and NOT an uplift measurement — that is the A/B engagement, which "
                 "additionally needs the tools image and an LLM proposer."),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreached_command_execution={result['reached_command_execution']} "
          f"reached_root={result['reached_root']} read_files={result['read_files']}")
    print(f"audit -> outputs/stage2_live_smoke_audit.jsonl\nreport -> {args.report_output}")


if __name__ == "__main__":
    main()
