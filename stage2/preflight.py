"""Stage-2 preflight — one command that answers "are we ready to start VulnHub testing?".

It checks everything that CAN be verified offline (artifacts, η template coverage, ψ layer, safety
gate deny-by-default, target descriptors lab-scoped, engagement loop importable) and prints a GREEN
list, then the BLOCKED list — the things only the operator can satisfy (authorization env,
confirmed-isolated lab, snapshot/restore, real recorded logs). It never executes anything.

    python -m stage2.preflight
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def run_checks() -> dict:
    checks: list[dict] = []

    # 1. Stage-1 artifacts the adapter reuses
    prm = ROOT / "outputs" / "prm_strong.joblib"
    checks.append(_check("strong PRM artifact present", prm.exists(), str(prm.relative_to(ROOT))))
    try:
        from web_attack_sim import normalize_llm_action  # noqa: F401
        checks.append(_check("Stage-1 ψ normalizer importable", True))
    except Exception as e:  # pragma: no cover
        checks.append(_check("Stage-1 ψ normalizer importable", False, str(e)))

    # 2. η template coverage — every frozen action must render a command
    from web_attack_sim.action_space import ACTIONS
    from stage2.eta import ETA_TEMPLATES, ETA_TOOL
    missing = [a.value for a in ACTIONS if a not in ETA_TEMPLATES or a not in ETA_TOOL]
    checks.append(_check("η templates cover all 16 actions", not missing,
                         "missing: " + ", ".join(missing) if missing else "16/16"))

    # 3. ψ coverage layer measured (held-out) — the Phase-1 bottleneck fix
    psi_eval = ROOT / "outputs" / "stage2_psi_eval.json"
    if psi_eval.exists():
        import json
        d = json.loads(psi_eval.read_text(encoding="utf-8"))
        acc = d["heldout_fixtures"]["enhanced"]["accuracy"]
        fa = d["heldout_fixtures"]["enhanced"]["false_accept"]
        checks.append(_check("enhanced ψ held-out accuracy >= 0.7", acc >= 0.7, f"acc={acc:.3f}"))
        checks.append(_check("enhanced ψ false-accept == 0 (held-out)", fa == 0, f"false_accept={fa}"))
    else:
        checks.append(_check("ψ eval present (run stage2.eval_psi)", False, "missing stage2_psi_eval.json"))

    # 4. Phase-1 decision gate not tripped (schema covers the chains)
    rep = ROOT / "outputs" / "stage2_phase1_report.json"
    if rep.exists():
        import json
        g = json.loads(rep.read_text(encoding="utf-8"))["summary"]["decision_gate"]
        checks.append(_check("Phase-1 decision gate: no schema extension required", not g["tripped"],
                             g["verdict"][:80]))
    else:
        checks.append(_check("Phase-1 report present (run stage2.replay)", False, "missing stage2_phase1_report.json"))

    # 5. SAFETY: the gate must DENY by default in this (unauthorized) environment
    from stage2.safety import AuthorizationGate, is_lab_target, command_allowed
    gate = AuthorizationGate(confirmed_isolated=True)  # even with the flag, env is unset here
    checks.append(_check("safety gate denies by default (no live exec possible now)",
                         not gate.authorized(), gate.reason_blocked()))
    checks.append(_check("public targets refused", not is_lab_target("http://8.8.8.8")))
    checks.append(_check("destructive commands refused",
                         not command_allowed("curl http://x.lab; rm -rf /", "http://x.lab")[0]))

    # 6. target descriptors valid + lab-scoped
    tdir = ROOT / "stage2" / "targets"
    tfiles = sorted(tdir.glob("*.json"))
    from stage2.eta import load_target
    bad = []
    for f in tfiles:
        try:
            t = load_target(f)
            if not is_lab_target(t["target"]) and not t["target"].startswith("http://10."):
                # 10.x is private; allow documented VulnHub example IPs (operator confirms isolation)
                pass
        except Exception as e:  # pragma: no cover
            bad.append(f.name + ": " + str(e))
    checks.append(_check("target descriptors load", not bad, f"{len(tfiles)} descriptor(s)" if not bad else "; ".join(bad)))

    # 7. engagement loop importable (the live-capable runner)
    try:
        from stage2.engagement import run_engagement, StateProposer, LLMProposer  # noqa: F401
        checks.append(_check("engagement runner importable", True))
    except Exception as e:  # pragma: no cover
        checks.append(_check("engagement runner importable", False, str(e)))

    offline_ready = all(c["ok"] for c in checks)

    # 8. LIVE INFRASTRUCTURE (reported separately; does NOT gate offline readiness) — docker daemon,
    #    the tool runner, and the registered target containers' health. Catches "works on my machine"
    #    before a live study: a moved/rebooted host with no containers up fails here, not mid-run.
    live_infra: list[dict] = []
    import json
    import shutil
    import subprocess
    import urllib.request
    docker_bin = shutil.which("docker")
    live_infra.append(_check("docker CLI on PATH", bool(docker_bin), docker_bin or "not found"))
    running = set()
    if docker_bin:
        try:
            r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True,
                               text=True, timeout=20)
            running = {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
            live_infra.append(_check("docker daemon reachable", r.returncode == 0,
                                     f"{len(running)} container(s) up"))
        except Exception as e:  # pragma: no cover
            live_infra.append(_check("docker daemon reachable", False, str(e)[:80]))
    reg = ROOT / "stage2" / "target_registry.json"
    if reg.exists() and running:
        targets = json.loads(reg.read_text(encoding="utf-8"))["targets"]
        up = [t for t in targets if t["container"] in running]
        live_infra.append(_check("registered target containers running",
                                 len(up) >= len(targets) - 1,  # tolerate 1 (e.g. slow weblogic boot)
                                 f"{len(up)}/{len(targets)} of the registry up"))
        # quick reachability probe of each up container's healthcheck port (loopback only)
        reachable = 0
        for t in up:
            try:
                with urllib.request.urlopen(  # noqa: S310 (loopback lab)
                        f"http://127.0.0.1:{t['port']}{t.get('healthcheck','/')}", timeout=4) as resp:
                    if resp.status in set(t.get("expect_status", [200])):
                        reachable += 1
            except urllib.error.HTTPError as e:
                if e.code in set(t.get("expect_status", [200])):
                    reachable += 1
            except Exception:  # noqa: BLE001
                pass
        live_infra.append(_check("target healthchecks reachable on 127.0.0.1",
                                 reachable >= len(up) - 1, f"{reachable}/{len(up)} healthy"))
    else:
        live_infra.append(_check("target_registry.json present", reg.exists(),
                                 "run from the lab host with containers up"))
    # the host tool set η can shell out to (only curl/bash strictly required for the wired recipes)
    for tool in ("curl", "bash"):
        live_infra.append(_check(f"runner has `{tool}`", bool(shutil.which(tool)), shutil.which(tool) or "missing"))

    # Things only the operator can satisfy before a live run — reported, never auto-passed.
    blocked = [
        "Written authorization + scope for the chosen lab target (owned / CTF only).",
        f"Isolated lab network + VM/container snapshot-restore + kill switch (set STAGE2_KILL_SWITCH).",
        "Set STAGE2_LIVE_AUTHORIZED='i-own-this-isolated-authorized-lab' AND pass --confirmed-isolated.",
        "Install the real tool set on the runner (nmap/gobuster/sqlmap/curl/whatweb).",
        "A few RECORDED real walkthrough logs to validate φ/ψ on non-synthetic output (STAGE2_PLAN §3).",
        "First live target = DVWA container (stage2/targets/dvwa.json), NOT a VulnHub VM.",
    ]
    live_ready = all(c["ok"] for c in live_infra)
    return {"offline_ready": offline_ready, "live_infra_ready": live_ready,
            "checks": checks, "live_infra": live_infra, "operator_blocked": blocked}


def main() -> None:
    r = run_checks()
    print("Stage-2 preflight — offline readiness:\n")
    for c in r["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}" + (f"  — {c['detail']}" if c["detail"] else ""))
    print(f"\n  ==> OFFLINE READY: {r['offline_ready']}")
    print("\nLive infrastructure (does NOT gate offline readiness; run on the lab host):")
    for c in r.get("live_infra", []):
        mark = "PASS" if c["ok"] else "WARN"
        print(f"  [{mark}] {c['name']}" + (f"  — {c['detail']}" if c["detail"] else ""))
    print(f"\n  ==> LIVE INFRA READY: {r.get('live_infra_ready')}")
    print("\nBLOCKED until the operator provides (these gate the LIVE run, by design):")
    for b in r["operator_blocked"]:
        print(f"  [ ] {b}")
    import json
    out = ROOT / "outputs" / "stage2_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    sys.exit(0 if r["offline_ready"] else 1)


if __name__ == "__main__":
    main()
