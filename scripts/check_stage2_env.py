"""Pre-flight checks for Stage-2 adapter experiments.

This script is intentionally conservative:
- offline Phase 1 is considered ready when Python deps, Stage-1 artifacts, fixtures,
  and the local Vulhub target are available;
- live Phase 2 remains not ready until a sandboxed LiveExecutor is implemented and
  the tool image/API key/authorization are explicitly present.

Run from WebAttackSim:
    python scripts/check_stage2_env.py
    python scripts/check_stage2_env.py --with-replay
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPOSE_DIR = Path(os.environ.get("STAGE2_VULHUB_COMPOSE_DIR", r"E:\PT\vulhub\thinkphp\5-rce"))
DEFAULT_TARGET_URL = os.environ.get("STAGE2_TARGET_URL", "http://127.0.0.1:8080/")
DEFAULT_DOCKER_NETWORK = os.environ.get("STAGE2_DOCKER_NETWORK", "5-rce_default")
TOOLS_IMAGE = os.environ.get("STAGE2_TOOLS_IMAGE", "webattacksim-stage2-tools")


def _cmd(args: list[str], *, cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd or ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {timeout}s",
        }


def _row(name: str, status: str, detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


def check_python() -> list[dict[str, Any]]:
    rows = [_row("python", "ok", sys.version.split()[0], executable=sys.executable)]
    modules = {
        "joblib": "joblib",
        "numpy": "numpy",
        "pytest": "pytest",
        "requests": "requests",
        "scikit-learn": "sklearn",
        "torch": "torch",
    }
    for label, module in modules.items():
        try:
            importlib.import_module(module)
            rows.append(_row(f"python package: {label}", "ok"))
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic script
            rows.append(_row(f"python package: {label}", "fail", str(exc)))
    return rows


def check_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = [
        ROOT / "outputs" / "prm_strong.joblib",
        ROOT / "outputs" / "oracle_seed_gate.json",
        ROOT / "stage2" / "walkthroughs",
    ]
    for path in required:
        rows.append(_row(f"path exists: {path.relative_to(ROOT)}", "ok" if path.exists() else "fail", str(path)))

    gate_path = ROOT / "outputs" / "oracle_seed_gate.json"
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            rows.append(_row("oracle seed gate passed", "ok" if gate.get("passed") else "fail", str(gate.get("passed"))))
            ckpt = gate.get("canonical_checkpoint")
            if ckpt:
                ckpt_path = ROOT / ckpt
                rows.append(_row("canonical oracle checkpoint", "ok" if ckpt_path.exists() else "fail", str(ckpt_path)))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row("oracle seed gate parse", "fail", str(exc)))

    fixtures = sorted((ROOT / "stage2" / "walkthroughs").glob("*.json"))
    rows.append(_row("walkthrough fixture count", "ok" if fixtures else "fail", str(len(fixtures))))
    return rows


def _http_status(url: str) -> dict[str, Any]:
    try:
        req = Request(url, headers={"User-Agent": "WebAttackSim-stage2-env-check"})
        with urlopen(req, timeout=10) as resp:  # noqa: S310 - local lab URL by design
            body = resp.read(160)
            return {"ok": 200 <= resp.status < 400, "status": resp.status, "sample": body.decode("utf-8", "ignore")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": None, "sample": str(exc)}


def check_docker(compose_dir: Path, target_url: str, network: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    docker = _cmd(["docker", "version", "--format", "{{.Server.Version}}"])
    rows.append(_row("docker engine", "ok" if docker["ok"] else "fail", docker["stdout"] or docker["stderr"]))
    compose = _cmd(["docker", "compose", "version"])
    rows.append(_row("docker compose", "ok" if compose["ok"] else "fail", compose["stdout"] or compose["stderr"]))

    compose_file = compose_dir / "docker-compose.yml"
    if compose_file.exists():
        text = compose_file.read_text(encoding="utf-8")
        localhost_only = "127.0.0.1:8080:80" in text
        rows.append(_row("vulhub compose file", "ok", str(compose_file)))
        rows.append(_row("vulhub host port bound to localhost", "ok" if localhost_only else "fail", "expected 127.0.0.1:8080:80"))
    else:
        rows.append(_row("vulhub compose file", "fail", str(compose_file)))

    ps = _cmd(["docker", "ps", "--filter", "name=5-rce-web-1", "--format", "{{.Names}} {{.Ports}}"])
    rows.append(_row("thinkphp container running", "ok" if ps["ok"] and "5-rce-web-1" in ps["stdout"] else "fail", ps["stdout"] or ps["stderr"]))
    inspect = _cmd(["docker", "inspect", "5-rce-web-1", "--format", "{{json .NetworkSettings.Ports}}"])
    rows.append(_row("thinkphp docker port inspect", "ok" if inspect["ok"] and "127.0.0.1" in inspect["stdout"] else "fail", inspect["stdout"] or inspect["stderr"]))
    net = _cmd(["docker", "network", "inspect", network, "--format", "{{.Name}}"])
    rows.append(_row("docker lab network", "ok" if net["ok"] else "fail", net["stdout"] or net["stderr"]))

    http = _http_status(target_url)
    rows.append(_row("target HTTP reachable from host", "ok" if http["ok"] else "fail", f"{target_url} -> {http['status']}", sample=http["sample"][:120]))
    return rows


def check_tools(network: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image = _cmd(["docker", "image", "inspect", TOOLS_IMAGE, "--format", "{{.Id}}"])
    rows.append(_row("stage2 Docker tool image", "ok" if image["ok"] else "warn", image["stdout"] or "not built yet"))
    if image["ok"]:
        smoke = _cmd(["docker", "run", "--rm", "--network", network, TOOLS_IMAGE, "curl", "-sI", "http://web/"], timeout=30)
        rows.append(_row("tool image can reach target as http://web/", "ok" if smoke["ok"] and "200" in smoke["stdout"] else "fail", smoke["stdout"] or smoke["stderr"]))

    host_tools = ["curl", "nmap", "gobuster", "ffuf", "sqlmap", "whatweb", "nikto"]
    for tool in host_tools:
        ps = (
            f"$c=Get-Command {tool} -ErrorAction SilentlyContinue; "
            "if ($c) { if ($c.Source) { $c.Source } else { $c.CommandType.ToString() } }"
        )
        probe = _cmd(["powershell", "-NoProfile", "-Command", ps])
        status = "ok" if probe["ok"] and probe["stdout"] else "warn"
        rows.append(_row(f"host PATH tool: {tool}", status, probe["stdout"] or "not on PATH"))
    return rows


def check_env_vars() -> list[dict[str, Any]]:
    rows = []
    rows.append(_row("DEEPSEEK_API_KEY", "ok" if os.environ.get("DEEPSEEK_API_KEY") else "warn",
                     "set" if os.environ.get("DEEPSEEK_API_KEY") else "not set; offline replay does not need it"))
    live = os.environ.get("STAGE2_LIVE_AUTHORIZED")
    if live:
        rows.append(_row("STAGE2_LIVE_AUTHORIZED", "warn", "set; make sure this is intentional"))
    else:
        rows.append(_row("STAGE2_LIVE_AUTHORIZED", "ok", "not set; live execution cannot start accidentally"))
    return rows


def check_live_executor() -> list[dict[str, Any]]:
    """Verify the LiveExecutor is the functional, gated implementation — by importing it and
    asserting it refuses without authorization, NOT by grepping the source for a string."""
    rows: list[dict[str, Any]] = []
    try:
        sys.path.insert(0, str(ROOT))
        from stage2.eta import LiveExecutor
        from web_attack_sim.action_space import ActionType
        from web_attack_sim.schemas import Action
        # functional = it executes (no NotImplementedError) but the gate refuses without auth here
        ex = LiveExecutor("http://dvwa.lab", confirmed_isolated=True)
        refused = False
        try:
            ex.run(Action(action_type=ActionType.SERVICE_ENUMERATION))
        except PermissionError:
            refused = True  # functional + correctly gated
        except NotImplementedError:
            return [_row("LiveExecutor implemented", "warn", "still a safety-gated stub (NotImplementedError)")]
        rows.append(_row("LiveExecutor implemented", "ok" if refused else "warn",
                         "functional + gated (refuses without authorization)" if refused
                         else "runs without refusing — check the safety gate"))
    except Exception as exc:  # noqa: BLE001
        rows.append(_row("LiveExecutor implemented", "warn", f"import failed: {exc}"))
    return rows


def run_replay() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_out = ROOT / "outputs" / "stage2_phase1_report_envcheck_enhanced.json"
    loop_out = ROOT / "outputs" / "stage2_closed_loop_envcheck.json"
    replay = _cmd([
        sys.executable, "-m", "stage2.replay", "--enhanced-psi",
        "--walkthroughs", "stage2/walkthroughs",
        "--report-output", str(replay_out),
    ], timeout=120)
    rows.append(_row("enhanced Phase-1 replay", "ok" if replay["ok"] else "fail", replay["stdout"] or replay["stderr"]))
    loop = _cmd([
        sys.executable, "-m", "stage2.closed_loop",
        "--walkthroughs", "stage2/walkthroughs",
        "--report-output", str(loop_out),
    ], timeout=120)
    rows.append(_row("offline closed-loop replay", "ok" if loop["ok"] else "fail", loop["stdout"] or loop["stderr"]))
    return rows


def summarize(sections: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    flat = [row for rows in sections.values() for row in rows]
    failures = [row for row in flat if row["status"] == "fail"]
    live_blockers = [
        row for row in flat
        if row["name"] in {"stage2 Docker tool image", "DEEPSEEK_API_KEY", "LiveExecutor implemented"}
        and row["status"] != "ok"
    ]
    return {
        "offline_phase1_ready": not failures,
        "live_phase2_ready": not failures and not live_blockers,
        "failure_count": len(failures),
        "live_blocker_count": len(live_blockers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Stage-2 experiment environment readiness.")
    parser.add_argument("--compose-dir", type=Path, default=DEFAULT_COMPOSE_DIR)
    parser.add_argument("--target-url", default=DEFAULT_TARGET_URL)
    parser.add_argument("--docker-network", default=DEFAULT_DOCKER_NETWORK)
    parser.add_argument("--with-replay", action="store_true", help="also run enhanced offline replay checks")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stage2_env_check.json")
    args = parser.parse_args()

    sections = {
        "python": check_python(),
        "artifacts": check_artifacts(),
        "docker_lab": check_docker(args.compose_dir, args.target_url, args.docker_network),
        "tools": check_tools(args.docker_network),
        "environment_variables": check_env_vars(),
        "live_executor": check_live_executor(),
    }
    if args.with_replay:
        sections["replay"] = run_replay()

    report = {
        "summary": summarize(sections),
        "config": {
            "root": str(ROOT),
            "compose_dir": str(args.compose_dir),
            "target_url": args.target_url,
            "docker_network": args.docker_network,
            "tools_image": TOOLS_IMAGE,
        },
        "sections": sections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for name, rows in sections.items():
        print(f"\n[{name}]")
        for row in rows:
            print(f"  {row['status'].upper():4s} {row['name']}: {row.get('detail', '')}")
    print(f"\nreport -> {args.output}")


if __name__ == "__main__":
    main()
