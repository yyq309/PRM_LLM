# Stage 2 Environment Status

Last checked: 2026-06-17

## Local lab target

- Vulhub checkout: `E:\PT\vulhub`
- Active target: `E:\PT\vulhub\thinkphp\5-rce`
- Host URL: `http://127.0.0.1:8080/`
- Docker container: `5-rce-web-1`
- Docker network: `5-rce_default`
- Tool-container URL: `http://web/` when the tool container joins `5-rce_default`

The target port is bound to localhost only:

```yaml
ports:
  - "127.0.0.1:8080:80"
```

Start or restart the target:

```powershell
cd E:\PT\vulhub\thinkphp\5-rce
docker compose up -d
```

Stop and clean the target:

```powershell
cd E:\PT\vulhub\thinkphp\5-rce
docker compose down -v
```

## Readiness check

Run this from `WebAttackSim`:

```powershell
python scripts\check_stage2_env.py --with-replay
```

Current result:

- Offline Phase 1 readiness: ready.
- Live Phase 2 readiness: not ready yet.
- Critical failures: 0.

The generated machine-readable report is:

```text
outputs\stage2_env_check.json
```

## Offline Phase 1 commands

These are safe and do not execute against a live target:

```powershell
python -m stage2.validate_fixture stage2\walkthroughs\dc-1.json
python -m stage2.replay --enhanced-psi --walkthroughs stage2\walkthroughs --report-output outputs\stage2_phase1_report_envcheck_enhanced.json
python -m stage2.closed_loop --walkthroughs stage2\walkthroughs --report-output outputs\stage2_closed_loop_envcheck.json
```

Latest checked metrics:

- Out-of-abstraction rate: 8.45%.
- Enhanced psi accuracy on in-abstraction steps: 78.46%.
- Enhanced psi false-accept on out-of-abstraction steps: 0.
- Phi field recall: 94.85%.
- Offline closed-loop chain adherence: 38.41%.

## Python dependencies

The minimal dependency list is in:

```text
requirements.txt
```

Install in a fresh environment:

```powershell
python -m pip install -r requirements.txt
```

## Stage 2 tool container

The tool image definition is in:

```text
stage2\tools\Dockerfile
```

Build command:

```powershell
docker build -t webattacksim-stage2-tools -f stage2\tools\Dockerfile .
```

Wrapper command:

```powershell
.\scripts\stage2_tool.ps1 curl -sI http://web/
```

Current note: the Dockerfile is present, but the image was not built because Docker Hub base-image downloads ended with EOF network errors. Retry the build when the network is stable.

## Safety gates

- `STAGE2_LIVE_AUTHORIZED` is intentionally unset, so live execution cannot start accidentally.
- `DEEPSEEK_API_KEY` is not set. Offline replay does not need it; live LLM candidate generation does.
- `stage2.eta.LiveExecutor` is still a safety-gated stub and raises `NotImplementedError` after authorization. A sandboxed command runner is still required before live Phase 2/3.

Do not run live exploitation outside owned, isolated, authorized Docker/VM labs.
