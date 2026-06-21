"""Helpers for wiring full-VM VulnHub targets into the Stage-2 lab.

This module is intentionally non-destructive: it never starts, stops, or snapshots VMs.
It only:

1. reads the operator-maintained `stage2/vm_target_registry.json`
2. renders hosts-file entries for the enabled VMware targets
3. probes the targets over HTTP from the host-only segment
4. syncs the chosen target URLs into `stage2/targets/*.json`

Typical flow:

    python -m stage2.vm_lab --list
    python -m stage2.vm_lab --render-hosts
    python -m stage2.vm_lab --check
    python -m stage2.vm_lab --sync-targets
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ipaddress
import json
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse
import urllib.error
import urllib.request

from stage2.safety import is_lab_target, target_host


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "stage2" / "vm_target_registry.json"
DEFAULT_REPORT = ROOT / "outputs" / "stage2_vm_lab_check.json"


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _load_registry(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("targets"), list):
        raise SystemExit(f"{path} must contain a top-level 'targets' list")
    return data


def _iter_targets(data: dict, *, include_disabled: bool = False, label: str | None = None) -> list[dict]:
    selected = []
    for entry in data.get("targets", []):
        if label and entry.get("label") != label:
            continue
        if not include_disabled and not entry.get("enabled", True):
            continue
        selected.append(entry)
    return selected


def _descriptor_path(entry: dict) -> Path:
    return ROOT / "stage2" / entry["descriptor"]


def _is_private_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def _resolve_host(host: str) -> tuple[bool, str]:
    try:
        return True, socket.gethostbyname(host)
    except OSError as e:
        return False, str(e)


def _probe_url(entry: dict) -> str:
    if entry.get("probe_url"):
        return str(entry["probe_url"])
    scheme = entry.get("probe_scheme", "http")
    host = entry["ip"]
    port = entry.get("port")
    path = entry.get("healthcheck", "/")
    netloc = f"{host}:{port}" if port else host
    return f"{scheme}://{netloc}{path}"


def _expected_host(entry: dict) -> str | None:
    explicit = entry.get("http_host")
    if explicit:
        return str(explicit)
    target = entry.get("target", "")
    host = target_host(target)
    if host and host != entry.get("ip"):
        return host
    return None


def _probe(entry: dict, timeout: int = 5) -> tuple[bool, int | str, str]:
    url = _probe_url(entry)
    headers = {}
    host_header = _expected_host(entry)
    if host_header:
        headers["Host"] = host_header
    req = urllib.request.Request(url, headers=headers, method="GET")
    expect = set(entry.get("expect_status", [200]))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - lab-only probe
            ok = resp.status in expect
            return ok, resp.status, url
    except urllib.error.HTTPError as e:
        ok = e.code in expect
        return ok, e.code, url
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__, url


def render_hosts(entries: list[dict]) -> str:
    lines = [
        "# Add these lines to the attacker and host OS hosts file if you use hostname-based targets.",
        "# Windows hosts path: C:\\Windows\\System32\\drivers\\etc\\hosts",
    ]
    for entry in entries:
        hostnames = []
        for key in ("hostname",):
            value = entry.get(key)
            if value and value not in hostnames:
                hostnames.append(value)
        target = entry.get("target")
        parsed = target_host(target or "")
        if parsed and parsed not in hostnames and parsed != entry.get("ip"):
            hostnames.append(parsed)
        if hostnames:
            lines.append(f"{entry['ip']} " + " ".join(hostnames))
    return "\n".join(lines)


def list_targets(entries: list[dict]) -> str:
    rows = []
    for entry in entries:
        status = "enabled" if entry.get("enabled", True) else "disabled"
        rows.append(
            f"{entry['label']:12s} [{status}] {entry['target']:24s} -> {entry['descriptor']}"
        )
    return "\n".join(rows)


def check_targets(entries: list[dict]) -> dict:
    results = []
    for entry in entries:
        descriptor = _descriptor_path(entry)
        target = entry["target"]
        checks = [
            _check("ip is private/loopback", _is_private_ip(entry["ip"]), entry["ip"]),
            _check("target is lab-scoped", is_lab_target(target), target),
            _check("descriptor exists", descriptor.exists(), str(descriptor.relative_to(ROOT)) if descriptor.exists() else str(descriptor)),
        ]
        host = target_host(target)
        if host and host != entry["ip"]:
            ok_resolve, detail = _resolve_host(host)
            if ok_resolve:
                checks.append(_check("target hostname resolves", detail == entry["ip"], f"{host} -> {detail}"))
            else:
                checks.append(_check("target hostname resolves", False, f"{host}: {detail}"))
        ok_probe, status, url = _probe(entry)
        checks.append(_check("http healthcheck reachable", ok_probe, f"{url} -> {status}"))
        results.append({
            "label": entry["label"],
            "target": target,
            "descriptor": entry["descriptor"],
            "ok": all(c["ok"] for c in checks),
            "checks": checks,
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": results,
        "all_ok": all(r["ok"] for r in results) if results else False,
    }


def sync_targets(entries: list[dict]) -> list[dict]:
    changed = []
    for entry in entries:
        path = _descriptor_path(entry)
        if not path.exists():
            changed.append({
                "label": entry["label"],
                "ok": False,
                "detail": f"descriptor missing: {path}",
            })
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        before = data.get("target")
        after = entry["target"]
        if before != after:
            data["target"] = after
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed.append({
            "label": entry["label"],
            "ok": True,
            "before": before,
            "after": after,
            "changed": before != after,
            "descriptor": str(path.relative_to(ROOT)),
        })
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Helpers for a local VMware VulnHub lab registry.")
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--label", help="Operate on a single label from vm_target_registry.json")
    ap.add_argument("--include-disabled", action="store_true", help="Include entries with enabled=false")
    ap.add_argument("--list", action="store_true", help="List the selected VM targets")
    ap.add_argument("--render-hosts", action="store_true", help="Print hosts-file lines for the selected targets")
    ap.add_argument("--check", action="store_true", help="Run descriptor/hostname/HTTP health checks")
    ap.add_argument("--sync-targets", action="store_true", help="Write each selected registry target URL into its descriptor JSON")
    ap.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    data = _load_registry(args.registry)
    entries = _iter_targets(data, include_disabled=args.include_disabled, label=args.label)
    if not entries:
        raise SystemExit("No matching targets found in vm_target_registry.json")

    if args.list:
        print(list_targets(entries))
    if args.render_hosts:
        print(render_hosts(entries))
    if args.check:
        report = check_targets(entries)
        for target in report["targets"]:
            flag = "PASS" if target["ok"] else "WARN"
            print(f"[{flag}] {target['label']} -> {target['target']}")
            for item in target["checks"]:
                mark = "PASS" if item["ok"] else "WARN"
                detail = f" :: {item['detail']}" if item["detail"] else ""
                print(f"  [{mark}] {item['name']}{detail}")
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report_output}")
    if args.sync_targets:
        results = sync_targets(entries)
        for item in results:
            if not item["ok"]:
                print(f"[WARN] {item['label']} :: {item['detail']}")
                continue
            action = "updated" if item["changed"] else "unchanged"
            print(f"[{action}] {item['label']} :: {item['before']} -> {item['after']} ({item['descriptor']})")
    if not any((args.list, args.render_hosts, args.check, args.sync_targets)):
        ap.print_help()


if __name__ == "__main__":
    main()
