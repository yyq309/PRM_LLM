# Stage 2 — Phase 2/3 Runbook (going live, safely)

Everything offline is built and green (`python -m stage2.preflight`). This runbook is the exact
sequence to start **live** testing once you have an authorized, isolated lab. Nothing here runs by
itself — every live step is gated by `stage2/safety.py`.

> **Hard rule.** Live execution only against an **owned / CTF / authorized**, **network-isolated**,
> **snapshot-restorable** target. The gate refuses public IPs and public hostnames; do not try to
> bypass it.

## 0. Preconditions (operator)

- [ ] Written authorization + scope for the target (owned or CTF box).
- [ ] Isolated virtual network (VirtualBox host-only / internal vSwitch / a private Docker bridge).
      No route to the internet or other hosts.
- [ ] Snapshot the target VM/container so you can restore after each run.
- [ ] A kill switch: pick a path and `export STAGE2_KILL_SWITCH=/tmp/stage2.stop`. `touch` it to
      halt all live execution immediately.
- [ ] Tools installed on the runner: `nmap gobuster sqlmap curl whatweb` (η allow-list).
- [ ] (Optional, recommended) a few **recorded real walkthrough logs** dropped into a folder so you
      can validate φ/ψ on non-synthetic output before driving the loop (see §2).

## 1. Confirm offline readiness

```powershell
python -m stage2.preflight          # must print OFFLINE READY: True
```

## 2. (Recommended) validate φ/ψ on REAL recorded logs first

The Phase-1 numbers (out-of-abstraction 8.5%, ψ 78.5%, φ 94.8%) are on author-constructed fixtures.
Before spending live effort, drop a handful of **real** captured walkthroughs (same JSON schema as
`stage2/walkthroughs/*.json`, `source` marked as a live capture) into `stage2/real_logs/` and run:

```powershell
python -m stage2.replay  --walkthroughs stage2\real_logs --enhanced-psi --report-output outputs\stage2_real_replay.json
python -m stage2.eval_psi --walkthroughs stage2\real_logs --report-output outputs\stage2_real_psi_eval.json
```

If ψ/φ hold up on real messy output → proceed. If they crater → fix the adapter before live (cheaper).

## 3. Phase 2 — single-box closed loop on **DVWA** (not VulnHub)

DVWA is a single web-app container that fits the abstraction; start here.

```powershell
# bring up DVWA on an isolated bridge, mapped to a .lab host, e.g. http://dvwa.lab
# edit stage2/targets/dvwa.json "target" if needed (must stay private/.lab)

$env:STAGE2_LIVE_AUTHORIZED = "i-own-this-isolated-authorized-lab"
$env:STAGE2_KILL_SWITCH      = "C:\tmp\stage2.stop"

python -m stage2.engagement `
  --target stage2\targets\dvwa.json `
  --executor live --proposer llm --mode ab `
  --confirmed-isolated `
  --budget 14 --report-output outputs\stage2_dvwa_live.json
```

- `--executor live` runs real commands **only** past the gate; every command is in
  `outputs/stage2_engagement_audit.jsonl`.
- `--proposer llm` uses DeepSeek (key from `$DEEPSEEK_API_KEY` only) to propose candidate steps.
- `--mode ab` runs both `prm` (PRM reranks the candidates) and `llm_only` (LLM order) so you get the
  uplift comparison.
- Restore the snapshot between runs.

**Success for Phase 2:** the loop reaches a flag / command-exec on DVWA with the PRM in the loop, and
the audit log is clean.

## 4. Phase 3 — A/B uplift across a small box set

Repeat §3 across DVWA, OWASP Juice Shop, and 1–2 **web-entry** VulnHub boxes
(`stage2/targets/vulnhub-dc-1.json` as a template — confirm the IP is private first). Run each box
several times per mode, then report **goal-reach rate / steps / wasted-actions, PRM vs LLM-only, with
confidence intervals**, alongside the out-of-abstraction rate and a failure taxonomy. Positive or
negative — report it honestly. That number is the Stage-2 deliverable.

## 5. Expectations (so the result is read correctly)

- The PRM is a **reranker, not a policy**. Frame the win as "PRM-rerank of the LLM's candidate
  next-steps beats LLM-only", not "PRM drives the box".
- Expect a **modest** uplift on the in-schema (~92%) portion; out-of-abstraction steps (SSH, offline
  crack, kernel/SMB) are handled by the operator, not the PRM.
- The frozen PRM consumes **training-time-ψ** features; the enhanced ψ drives action mapping/η. To put
  enhanced ψ into the PRM's feature path you must regenerate features and retrain the PRM (Stage-1
  retrain) — out of scope for Phase 2.

## 6. Safety stops

- `touch $STAGE2_KILL_SWITCH` halts the loop immediately (checked every iteration).
- The gate refuses any command whose target is not private/loopback/.lab, any binary off the
  allow-list, and any destructive token. Treat a refusal as a real signal, not an obstacle.
