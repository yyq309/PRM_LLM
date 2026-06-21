"""Reset / snapshot / healthcheck for the Stage-2 full-chain VulnHub VMs (VMware).

The VM analogue of stage2/reset_target.py (which is docker-compose only). Full-chain boxes mutate
state (Drupalgeddon2 drops files, privesc writes a fake binary / loads a UDF, etc.), so between
trials the VM is reverted to a 'clean' snapshot. Driven by stage2/vm_target_registry.json.

    python -m stage2.vm_reset --list
    python -m stage2.vm_reset --check                 # healthcheck every enabled VM (no revert)
    python -m stage2.vm_reset --label DC-1            # revertToSnapshot 'clean' && start && healthcheck
    python -m stage2.vm_reset --all                   # revert+start every enabled VM
    python -m stage2.vm_reset --label DC-1 --snapshot clean   # CREATE the 'clean' snapshot (after first good boot)

SAFETY: this only drives vmrun against the operator-listed local VMs. It performs NO exploitation and
never touches a target outside the registry. Flags are never read here.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "stage2" / "vm_target_registry.json"


def _load() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _vmrun_path(reg: dict) -> str:
    return reg.get("vmrun") or "vmrun"


def _vmrun(vmrun: str, *args: str, timeout: int = 180) -> tuple[bool, str]:
    """Run a vmrun subcommand; returns (ok, output)."""
    try:
        r = subprocess.run([vmrun, "-T", "ws", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return False, f"vmrun not found at {vmrun!r}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _healthcheck(t: dict, timeout: int = 180, interval: int = 4) -> tuple[bool, object]:
    """Poll http://<ip>:<port><healthcheck> until an expected status, or time out. VulnHub web boxes
    take a while to bring the web service up after a cold boot."""
    ip = t.get("ip")
    port = t.get("port", 80)
    url = f"http://{ip}:{port}{t.get('healthcheck', '/')}"
    expect = set(t.get("expect_status", [200, 301, 302, 403]))
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5) as r:  # noqa: S310 (lab only)
                if r.status in expect:
                    return True, r.status
                last = r.status
        except urllib.error.HTTPError as e:
            if e.code in expect:
                return True, e.code
            last = e.code
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
        time.sleep(interval)
    return False, last


def reset_one(t: dict, vmrun: str, *, check_only: bool = False, create_snapshot: str | None = None) -> dict:
    label = t["label"]
    vmx = t.get("vmx")
    snap = t.get("snapshot", "clean")
    if create_snapshot:
        ok, out = _vmrun(vmrun, "snapshot", vmx, create_snapshot)
        return {"label": label, "ok": ok, "mode": "snapshot", "detail": out[:200] if not ok else f"snapshot '{create_snapshot}' created"}
    if not check_only:
        if not vmx or not Path(vmx).exists():
            return {"label": label, "ok": False, "detail": f"vmx missing: {vmx}"}
        ok_rev, out_rev = _vmrun(vmrun, "revertToSnapshot", vmx, snap)
        if not ok_rev:
            return {"label": label, "ok": False, "mode": "reset",
                    "detail": f"revertToSnapshot '{snap}' failed (create it with --snapshot {snap}): {out_rev[:160]}"}
        ok_start, out_start = _vmrun(vmrun, "start", vmx, "nogui")
        if not ok_start and "already running" not in out_start.lower():
            return {"label": label, "ok": False, "mode": "reset", "detail": f"start failed: {out_start[:160]}"}
    healthy, status = _healthcheck(t, timeout=200 if not check_only else 12)
    return {"label": label, "ok": bool(healthy), "status": status, "ip": t.get("ip"),
            "mode": "check" if check_only else "reset"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Reset/snapshot/healthcheck Stage-2 full-chain VulnHub VMs.")
    ap.add_argument("--label", help="single VM label (see --list)")
    ap.add_argument("--all", action="store_true", help="every enabled VM")
    ap.add_argument("--check", action="store_true", help="healthcheck only; do not revert/start")
    ap.add_argument("--snapshot", metavar="NAME", help="CREATE a snapshot of the selected VM(s) with this name")
    ap.add_argument("--include-disabled", action="store_true", help="include enabled=false VMs")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    reg = _load()
    vmrun = _vmrun_path(reg)
    targets = [t for t in reg.get("targets", []) if args.include_disabled or t.get("enabled", True)]

    if args.list:
        for t in reg.get("targets", []):
            en = "enabled " if t.get("enabled", True) else "disabled"
            print(f"  {t['label']:12s} [{en}] {t.get('ip',''):16s} {t.get('privesc_vector',''):16s} -> {t.get('descriptor','')}")
        return

    sel = targets if args.all else [t for t in targets if t["label"] == args.label]
    if not sel:
        print("no VM selected; use --label <L> or --all (see --list)")
        return

    results = [reset_one(t, vmrun, check_only=args.check, create_snapshot=args.snapshot) for t in sel]
    for r in results:
        flag = "OK  " if r["ok"] else "DOWN"
        print(f"  [{flag}] {r['label']:12s} {r.get('ip',''):16s} status={r.get('status')} ({r.get('mode')})"
              + (f"  :: {r['detail']}" if r.get("detail") else ""))
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} ok")


if __name__ == "__main__":
    main()
