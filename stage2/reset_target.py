"""Unified reset / start / healthcheck for the Stage-2 live Vulhub targets.

Addresses the reproducibility + fairness gap: between trials (and between arms) a target should be
returned to a known-clean state so one run cannot pollute the next (e.g. a Tomcat PUT-webshell or a
Drupal/Gitea registration persisting into the other arm). Driven by stage2/target_registry.json.

    python -m stage2.reset_target --list
    python -m stage2.reset_target --label Tomcat-12615            # down -v && up -d && healthcheck
    python -m stage2.reset_target --label Tomcat-12615 --check    # healthcheck only (no restart)
    python -m stage2.reset_target --all --check                  # healthcheck every box

NOTE on cost: a full `down -v && up -d` is correct for STATE-MUTATING boxes (Tomcat upload, Drupal /
Gitea registration). For the read-only recon->RCE->`cat` boxes (ThinkPHP, Struts2, php-cgi, LFI,
Rails) no arm writes persistent state, so cross-arm contamination is ~nil and a reset is optional; the
registry marks which boxes mutate state via `reset_note`. WebLogic boots slowly — budget for it.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "stage2" / "target_registry.json"


def _load():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["targets"]


def _healthcheck(t, timeout=120, interval=3):
    """Poll http://127.0.0.1:<port><healthcheck> until an expected status, or time out."""
    url = f"http://127.0.0.1:{t['port']}{t.get('healthcheck', '/')}"
    expect = set(t.get("expect_status", [200]))
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310 (loopback lab only)
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


def _compose(cmd, cwd):
    """Run `docker compose <cmd>` in cwd; returns (ok, output)."""
    try:
        r = subprocess.run(["docker", "compose"] + cmd, cwd=cwd, capture_output=True,
                           text=True, timeout=300)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def reset_one(t, check_only=False):
    cd = t["compose_dir"]
    if not Path(cd).exists():
        return {"label": t["label"], "ok": False, "detail": f"compose_dir missing: {cd}"}
    if not check_only:
        ok_d, _ = _compose(["down", "-v"], cd)
        ok_u, out_u = _compose(["up", "-d"], cd)
        if not ok_u:
            return {"label": t["label"], "ok": False, "detail": f"up failed: {out_u[:200]}"}
    healthy, status = _healthcheck(t, timeout=140 if "weblogic" in t["label"].lower() else 90)
    return {"label": t["label"], "ok": bool(healthy), "status": status,
            "port": t["port"], "mode": "check" if check_only else "reset"}


def main():
    ap = argparse.ArgumentParser(description="Reset/start/healthcheck Stage-2 live targets.")
    ap.add_argument("--label", help="single target label (see --list)")
    ap.add_argument("--all", action="store_true", help="every registered target")
    ap.add_argument("--check", action="store_true", help="healthcheck only; do not restart")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    targets = _load()
    if args.list:
        for t in targets:
            mut = " [state-mutating]" if t.get("reset_note") else ""
            print(f"  {t['label']:20s} :{t['port']}  {t['image']:32s} {t['vuln_class']}{mut}")
        return
    sel = targets if args.all else [t for t in targets if t["label"] == args.label]
    if not sel:
        print("no target selected; use --label <L> or --all (see --list)")
        return
    results = [reset_one(t, check_only=args.check) for t in sel]
    for r in results:
        flag = "OK " if r["ok"] else "DOWN"
        print(f"  [{flag}] {r['label']:20s} :{r.get('port')}  status={r.get('status')}  ({r['mode']})")
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} healthy")


if __name__ == "__main__":
    main()
