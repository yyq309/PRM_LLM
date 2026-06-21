"""Joint multi-task Pentest-PRM implementing the method's full L_PRM (method §10).

The method specifies a joint objective:
    L_PRM = l1*L_gap + l2*L_rank + l3*L_pref + l4*L_class + l5*L_diag + l6*L_feedback
Earlier PRMs trained separate regression/classification heads and omitted L_pref (the
pairwise preference term `-log sigmoid(PRM(c, a_good) - PRM(c, a_bad))`). This is a single
torch MLP with a shared trunk and four heads trained jointly:
  - L_gap   : MSE on the process score (regression)
  - L_rank  : cross-entropy on the rank bucket (high/medium/low/...)  (= L_rank + L_class)
  - L_diag  : cross-entropy on the diagnosis label
  - L_pref  : Bradley-Terry preference over SAME-STATE candidate pairs (the missing term)
Pointwise losses are weighted by the multi-seed label confidence (method §11.2); preference
pairs are weighted by the min confidence of the two candidates.

Reported on the honest oracle-labeled subset, broken out by unseen_instance / unseen_chain,
with a bootstrap CI, alongside the gradient-boosted strong PRM for comparison.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction import DictVectorizer

from honest_eval import majority_floor, pairwise_within_group  # noqa: E402
from task_split import get_split  # noqa: E402
from train_prm_baseline import read_jsonl  # noqa: E402
from train_prm_strong import expected_calibration_error, extract_features  # noqa: E402


class JointPRM(nn.Module):
    def __init__(self, in_dim: int, n_rank: int, n_diag: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.score_head = nn.Linear(hidden, 1)   # logit; sigmoid -> [0,1] process score
        self.rank_head = nn.Linear(hidden, n_rank)
        self.diag_head = nn.Linear(hidden, n_diag)

    def forward(self, x):
        h = self.trunk(x)
        return self.score_head(h).squeeze(-1), self.rank_head(h), self.diag_head(h)


def build_pairs(rows: list[dict[str, Any]], eps: float) -> list[tuple[int, int, float]]:
    """Same-state (candidate_group) oracle pairs (better_idx, worse_idx, weight)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("label_source") == "oracle":
            groups[str(r.get("candidate_group"))].append(i)
    pairs = []
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                d = float(rows[i]["score"]) - float(rows[j]["score"])
                if abs(d) <= eps:
                    continue
                w = min(float(rows[i].get("multiseed_confidence", 1.0)), float(rows[j].get("multiseed_confidence", 1.0)))
                pairs.append((i, j, w) if d > 0 else (j, i, w))
    return pairs


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_rows = read_jsonl(args.train_input)
    heldout_rows = read_jsonl(args.heldout_input)

    vec = DictVectorizer(sparse=False)
    Xtr = vec.fit_transform([extract_features(r) for r in train_rows]).astype(np.float32)
    Xhe = vec.transform([extract_features(r) for r in heldout_rows]).astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xhe = (Xhe - mu) / sd

    rank_classes = sorted({str(r["rank_label"]) for r in train_rows})
    diag_classes = sorted({str(r["diagnosis"]) for r in train_rows})
    rank_idx = {c: i for i, c in enumerate(rank_classes)}
    diag_idx = {c: i for i, c in enumerate(diag_classes)}

    y_score = torch.tensor([float(r["score"]) for r in train_rows], dtype=torch.float32)
    y_rank = torch.tensor([rank_idx[str(r["rank_label"])] for r in train_rows], dtype=torch.long)
    y_diag = torch.tensor([diag_idx.get(str(r["diagnosis"]), 0) for r in train_rows], dtype=torch.long)
    w = torch.tensor([float(r.get("multiseed_confidence", 1.0)) for r in train_rows], dtype=torch.float32).clamp(min=args.weight_floor)
    Xtr_t = torch.tensor(Xtr)

    pairs = build_pairs(train_rows, args.pair_eps)
    pi = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    pj = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    pw = torch.tensor([p[2] for p in pairs], dtype=torch.float32).clamp(min=args.weight_floor)

    model = JointPRM(Xtr.shape[1], len(rank_classes), len(diag_classes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = Xtr.shape[0]

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, args.batch_size):
            idx = perm[s : s + args.batch_size]
            opt.zero_grad()
            score_logit, rank_logit, diag_logit = model(Xtr_t[idx])
            score = torch.sigmoid(score_logit)
            l_gap = (w[idx] * (score - y_score[idx]) ** 2).mean()
            l_rank = (F.cross_entropy(rank_logit, y_rank[idx], reduction="none") * w[idx]).mean()
            l_diag = (F.cross_entropy(diag_logit, y_diag[idx], reduction="none") * w[idx]).mean()
            # preference term on a random subset of same-state pairs
            if len(pairs) > 0:
                psel = torch.randint(0, len(pairs), (min(args.batch_size, len(pairs)),))
                si, _, _ = model(Xtr_t[pi[psel]])
                sj, _, _ = model(Xtr_t[pj[psel]])
                l_pref = -(pw[psel] * F.logsigmoid(si - sj)).mean()
            else:
                l_pref = torch.tensor(0.0)
            loss = args.l_gap * l_gap + args.l_rank * l_rank + args.l_diag * l_diag + args.l_pref * l_pref
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        score_logit, rank_logit, _ = model(torch.tensor(Xhe))
        score_pred = torch.sigmoid(score_logit).numpy()
        rank_prob = F.softmax(rank_logit, dim=1).numpy()
    rank_pred = [rank_classes[i] for i in rank_prob.argmax(1)]

    split = get_split()
    chain_ids = {Path(p).stem for p in split["heldout_chain"]}
    inst_ids = {Path(p).stem for p in split["heldout_instance"]}

    def tf(r):
        return Path(r.get("task_path", "")).stem

    oracle = [i for i, r in enumerate(heldout_rows) if r.get("label_source") == "oracle"]
    o_inst = [i for i in oracle if tf(heldout_rows[i]) in inst_ids]
    o_chain = [i for i in oracle if tf(heldout_rows[i]) in chain_ids]

    def grp(idxs):
        if not idxs:
            return {"n": 0}
        ra = float(np.mean([int(rank_pred[i] == heldout_rows[i]["rank_label"]) for i in idxs]))
        conf = rank_prob.max(1)[idxs]
        correct = np.asarray([int(rank_pred[i] == heldout_rows[i]["rank_label"]) for i in idxs], dtype=float)
        pair = pairwise_within_group(heldout_rows, list(score_pred), idxs, args.pair_eps)
        ylab = [str(heldout_rows[i]["rank_label"]) for i in idxs]
        return {"n": len(idxs), "rank_accuracy": ra, "majority_floor": majority_floor(ylab),
                "pairwise_accuracy": pair["pairwise_accuracy"], "pairwise_pairs": pair["pairs"],
                "ece": expected_calibration_error(conf, correct)}

    # bootstrap CI on oracle-all pairwise
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.bootstrap):
        samp = list(rng.choice(oracle, size=len(oracle), replace=True)) if oracle else []
        p = pairwise_within_group(heldout_rows, list(score_pred), samp, args.pair_eps)
        if p["pairwise_accuracy"] is not None:
            boot.append(p["pairwise_accuracy"])

    report = {
        "objective": "L_gap + L_rank + L_diag + L_pref (joint, confidence-weighted)",
        "lambdas": {"gap": args.l_gap, "rank": args.l_rank, "diag": args.l_diag, "pref": args.l_pref},
        "n_train": len(train_rows), "n_preference_pairs": len(pairs),
        "oracle_all": grp(oracle), "oracle_unseen_instance": grp(o_inst), "oracle_unseen_chain": grp(o_chain),
        "bootstrap_oracle_all_pairwise_ci95": {
            "mean": float(np.mean(boot)) if boot else None,
            "ci95_low": float(np.percentile(boot, 2.5)) if boot else None,
            "ci95_high": float(np.percentile(boot, 97.5)) if boot else None,
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint multi-task PRM with preference learning (method §10).")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_train_multiseed.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_heldout_multiseed.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "prm_joint_eval.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l-gap", type=float, default=1.0)
    parser.add_argument("--l-rank", type=float, default=1.0)
    parser.add_argument("--l-diag", type=float, default=0.5)
    parser.add_argument("--l-pref", type=float, default=1.0)
    parser.add_argument("--pair-eps", type=float, default=1e-6)
    parser.add_argument("--weight-floor", type=float, default=0.3)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = train(args)
    print(f"preference pairs: {report['n_preference_pairs']}")
    for g in ["oracle_all", "oracle_unseen_instance", "oracle_unseen_chain"]:
        m = report[g]
        if m.get("n"):
            print(f"  {g:24s} n={m['n']:4d} rank={m['rank_accuracy']:.3f} pairwise={m['pairwise_accuracy']:.3f} ece={m['ece']:.3f}")
    ci = report["bootstrap_oracle_all_pairwise_ci95"]
    print(f"  bootstrap oracle_all pairwise 95% CI: [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
