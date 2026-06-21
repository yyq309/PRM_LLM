# Real recorded walkthrough logs (drop-in)

Put **real, captured** single-host walkthroughs here (one JSON per box) to validate φ/ψ on
non-synthetic output before any live run. Same schema as `stage2/walkthroughs/*.json`
(see `stage2/fixtures.py`), with two differences that keep the measurement honest:

- `source` must say it is a **live capture** (e.g. asciinema / saved tool stdout), not constructed.
- Hand-label each step's `reference_abstract_action` (one of the 16 actions or `out_of_abstraction`)
  and `reference_state_after` from what actually happened — that is the ground truth φ/ψ are scored
  against.

Then:

```powershell
python -m stage2.replay  --walkthroughs stage2\real_logs --enhanced-psi --report-output outputs\stage2_real_replay.json
python -m stage2.eval_psi --walkthroughs stage2\real_logs --report-output outputs\stage2_real_psi_eval.json
```

This is the single highest-value de-risking step before live execution: the Phase-1 numbers are on
author-constructed fixtures, and real tool output is messier. If φ/ψ hold up here, proceed to the
Phase-2 runbook; if not, fix the adapter first (cheaper than learning it live).

This folder is intentionally empty in the repo — real logs are operator-provided.
