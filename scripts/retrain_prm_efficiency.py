"""Efficiency-corrected PRM retrain (③): fix the recon bias by AUGMENTING the training labels with
"recon-when-already-known -> low value" samples, then retraining ONLY the score regressor (reusing the
frozen vectorizer, so it is surgical and prm_strong stays intact).

Why: the masked abstract oracle rarely produced "recon after the info is already known" states (the mask
removes completed recon), so the PRM learned `web_path_enumeration` mean-label 0.887 and EXTRAPOLATES that
high value to real-target states where recon is wasteful -> it reranks a good LLM order toward recon. We
teach the contrast directly: the SAME recon action in an ADVANCED state (paths known / shell established /
vuln verified) is low value. This is a principled training-signal correction (like adding a step cost),
NOT inference-time hacking and NOT box-specific leakage.

    python scripts/retrain_prm_efficiency.py
-> outputs/prm_strong_v2.joblib  (drop-in for --prm-model)
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_prm_strong import extract_features  # noqa: E402

RECON_TYPES = {"http_fingerprint", "web_path_enumeration", "content_retrieval",
               "service_enumeration", "input_discovery"}


def _advance_context(ctx: str, shell: str) -> str:
    """Return the context as if recon were ALREADY done: paths known, (optionally) a foothold + verified
    vuln. extract_features reads these via num_paths / shell_state / num_verified_vulns."""
    ctx = re.sub(r"Known paths: \[[^\]]*\]", "Known paths: ['/admin', '/login', '/config', '/api', '/uploads']", ctx)
    ctx = re.sub(r"Verified vulnerabilities: \[[^\]]*\]", "Verified vulnerabilities: ['rce']", ctx)
    ctx = re.sub(r"Shell state: \w+", f"Shell state: {shell}", ctx)
    return ctx


def main() -> None:
    prm = joblib.load(ROOT / "outputs" / "prm_strong.joblib")
    vec = prm["vectorizer"]                       # REUSE the frozen vectorizer (surgical)
    rows = [json.loads(l) for l in open(ROOT / "outputs" / "prm_samples_train.jsonl", encoding="utf-8")]

    aug = []
    for r in rows:
        na = r.get("normalized_action") or {}
        atype = na.get("action_type")
        if atype not in RECON_TYPES:
            continue
        ctx = str(r.get("context", ""))
        # recon-when-advanced negatives at increasing advancement -> strongly decaying value
        for shell, low in (("none", 0.10), ("webshell", 0.05), ("command_execution", 0.02)):
            aug.append({**r, "context": _advance_context(ctx, shell), "score": low})

    all_rows = rows + aug
    # weight the augmented negatives higher so the model is forced to learn the recon-decay contrast
    w = np.asarray([1.0] * len(rows) + [3.0] * len(aug), dtype=np.float32)
    print(f"original={len(rows)}  augmented(recon-when-advanced)={len(aug)}  total={len(all_rows)}  (aug weight 3x)")
    X = vec.transform([extract_features(r) for r in all_rows]).toarray()
    y = np.asarray([float(r["score"]) for r in all_rows], dtype=np.float32)
    score = HistGradientBoostingRegressor(random_state=0, max_iter=400, learning_rate=0.08)
    score.fit(X, y, sample_weight=w)

    out = {**prm, "score": score, "kind": "strong_efficiency_v2",
           "note": "recon-bias-corrected: augmented with recon-when-advanced->low-value, score model retrained"}
    joblib.dump(out, ROOT / "outputs" / "prm_strong_v2.joblib")

    # sanity: does v2 now rank exploit above recon in an ADVANCED state?
    def sc(model, action_type, ctx):
        feats = extract_features({"context": ctx, "normalized_action": {"action_type": action_type,
                                  "status": "valid", "target": None, "parameter": None}, "normalizer_confidence": 0.85})
        return float(np.clip(model.predict(vec.transform([feats]).toarray()), 0, 1)[0])

    adv = ("Scenario x, step 3. Known paths: ['/admin','/login','/config']. Known forms: []. Known "
           "parameters: []. Credentials: []. Auth state: anonymous. Shell state: none. Verified "
           "vulnerabilities: ['rce']. Read files: []. Failed branches: {}. Remaining budget: 10.")
    print("\nADVANCED state (paths known, vuln verified) — score web_path_enum vs exploit:")
    for tag, model in (("OLD prm", prm["score"]), ("NEW v2 ", score)):
        wpe = sc(model, "web_path_enumeration", adv); exp = sc(model, "exploit_attempt", adv)
        print(f"  {tag}: web_path_enum={wpe:.3f}  exploit_attempt={exp:.3f}  -> "
              f"{'exploit wins' if exp > wpe else 'RECON still wins'}")
    print("\nsaved -> outputs/prm_strong_v2.joblib")


if __name__ == "__main__":
    main()
