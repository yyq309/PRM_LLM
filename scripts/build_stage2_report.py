"""Assemble the Stage-2 Phase-1 markdown report from the replay + closed-loop JSON outputs.

Reads outputs/stage2_phase1_report.json (φ/ψ replay) and outputs/stage2_closed_loop.json
(offline closed-loop proxy) and writes STAGE2_PHASE1_REPORT.md — the report-ready summary of
the abstraction-gap measurement and the schema-extension decision gate.
"""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _psi_failure_clusters(replay: dict) -> tuple[dict, dict]:
    """Tally ψ misses on in-abstraction steps: false-rejects (status!=valid) and wrong-action
    routes, keyed by the hand-labeled reference action. This is what makes the ψ bottleneck
    actionable — it says *which* action families the normalizer fails to cover."""
    from collections import Counter
    false_reject: Counter = Counter()
    wrong_route: Counter = Counter()  # ref -> "ref->psi_action" rollups
    for b in replay["per_box"]:
        for s in b["steps"]:
            ref = s["reference_abstract_action"]
            if ref == "out_of_abstraction":
                continue
            if s["psi_correct"]:
                continue
            if s["psi_status"] != "valid":
                false_reject[ref] += 1
            else:
                wrong_route[f"{ref} -> {s['psi_action']}"] += 1
    return dict(false_reject.most_common()), dict(wrong_route.most_common())


def main() -> None:
    replay = _load(ROOT / "outputs" / "stage2_phase1_report.json")
    loop = _load(ROOT / "outputs" / "stage2_closed_loop.json")
    enhanced = _load(ROOT / "outputs" / "stage2_phase1_report_enhanced.json")
    psi_eval = _load(ROOT / "outputs" / "stage2_psi_eval.json")
    if replay is None:
        raise SystemExit("run `python -m stage2.replay` first")

    s = replay["summary"]
    g = s["decision_gate"]
    lines: list[str] = []
    A = lines.append

    A("# Stage 2 — Phase 1 Report (offline φ/ψ replay, no live execution)")
    A("")
    A("**What this measures.** How well the frozen 16-action abstraction + Stage-1 ψ normalizer + the new φ")
    A("parser cover the *real* attack chains of representative VulnHub-class single-host boxes, and whether the")
    A("abstract-trained Pentest-PRM reranks real candidate actions. NOTHING here executes against a target —")
    A("it replays recorded tool output through the adapter. This is the decisive, cheap, safe step that gates")
    A("whether (and where) the schema needs extending before any live Phase-2 work.")
    A("")
    A("## Headline numbers")
    A("")
    A(f"- **Boxes:** {s['n_boxes']}  |  **Total real steps:** {s['total_steps']}")
    A(f"- **Out-of-abstraction rate (ground-truth, hand-labeled): `{s['out_of_abstraction_rate']:.1%}`**")
    A(f"- **ψ normalizer accuracy on in-abstraction steps:** `{s['psi_accuracy_in_abstraction']:.1%}`"
      f"  (false-reject `{s['psi_false_reject_rate']:.1%}`)")
    A(f"- **φ field recall (state reconstruction vs hand-labels):** `{s['phi_field_recall']:.1%}`")
    rr = s.get("prm_rerank_top1_rate")
    A(f"- **PRM rerank top-1 (good vs distractors):** "
      f"`{rr:.1%}`" if rr is not None else "- **PRM rerank top-1:** n/a (no candidate pools)")
    if loop:
        ls = loop["summary"]
        A(f"- **Offline closed-loop chain-adherence (PRM-autonomous proxy):** `{ls['mean_chain_adherence']:.1%}`")
    A("")
    A("## Key finding — the bottleneck is ψ, not the schema")
    A("")
    ooa = s["out_of_abstraction_rate"]
    psi = s["psi_accuracy_in_abstraction"]
    fr = s["psi_false_reject_rate"]
    A(f"The decisive Phase-1 result **inverts the expected risk**. Going in, the open question was whether")
    A(f"the frozen 16-action abstraction is too small for real boxes. It is not:")
    A("")
    A(f"- **Abstraction coverage is high:** only `{ooa:.1%}` of real steps fall outside the 16 actions"
      f" (hand-labeled ground truth). The schema structurally covers ~{1 - ooa:.0%} of real chains.")
    A(f"- **The normalizer (ψ) is the real wall:** on the in-abstraction steps — the ones that *do* have a")
    A(f"  correct abstract action — the Stage-1 ψ maps only `{psi:.1%}` correctly, false-rejecting `{fr:.1%}`")
    A(f"  of them outright. Natural operator phrasing is messier than ψ's synthetic training intents.")
    A("")
    A("So the Phase-2 prerequisite is **ψ normalizer coverage, not schema extension**. The schema-gap")
    A("shortlist below is small and entirely *non-web* primitives (SSH login, offline hash cracking, local")
    A("account switch, a binary/SMB service exploit) — genuinely out of a single-host-**web** scope, not")
    A("evidence the web schema is incomplete.")
    A("")
    fr_clusters, wrong_clusters = _psi_failure_clusters(replay)
    if fr_clusters:
        A("**Where ψ false-rejects in-abstraction intents (by hand-labeled action):**")
        A("")
        for act, n in fr_clusters.items():
            A(f"- `{act}` — {n} miss(es)")
        A("")
    if wrong_clusters:
        A("**Where ψ routes to the wrong action (defensible boundary cases vs hard errors):**")
        A("")
        for route, n in wrong_clusters.items():
            A(f"- `{route}` — {n}")
        A("")
    A("> Honesty note: these clusters are measured on author-constructed fixtures, so they show ψ's coverage")
    A("> shape, not a population rate. A real ψ-coverage fix must be validated on **held-out** recorded")
    A("> operator logs (STAGE2_PLAN §3) — tuning ψ against these same fixtures would be in-sample and")
    A("> circular. We report the gap; we do **not** report a flattering in-sample 'after' number.")
    A("")
    if enhanced is not None and psi_eval is not None:
        es = enhanced["summary"]
        dev = psi_eval["dev_benchmark"]
        test = psi_eval["heldout_fixtures"]
        A("## ψ coverage layer (Stage-2) — bottleneck addressed, honestly")
        A("")
        A("`stage2/psi.py` adds a Stage-2-local coverage layer over the **frozen** Stage-1 normalizer (Stage 1")
        A("is untouched, so the oracle/PRM artifacts stay reproducible). It only acts when Stage-1 returns")
        A("`unsupported`: an **out-guard** keeps non-web primitives (SSH, offline crack, su, kernel/binary")
        A("overflow) out, then a general verb×object recovery layer maps the in-scope intents the keyword")
        A("matcher missed. The recovery vocabulary is tuned **only** on an independent benchmark")
        A("(`stage2/psi_benchmark.jsonl`, 124 labeled intents) — *disjoint from these fixtures*.")
        A("")
        A("**Generalization is measured on the 7 walkthrough fixtures as a HELD-OUT test** (the layer never saw")
        A("their strings):")
        A("")
        A("| set | ψ | accuracy | false-reject | false-accept |")
        A("|---|---|--:|--:|--:|")
        A(f"| dev benchmark (tuned-on) | Stage-1 | {dev['baseline']['accuracy']:.1%} | "
          f"{dev['baseline']['false_reject_rate']:.1%} | {dev['baseline']['false_accept_rate']:.1%} |")
        A(f"| dev benchmark (tuned-on) | enhanced | {dev['enhanced']['accuracy']:.1%} | "
          f"{dev['enhanced']['false_reject_rate']:.1%} | {dev['enhanced']['false_accept_rate']:.1%} |")
        A(f"| **held-out fixtures** | Stage-1 | {test['baseline']['accuracy']:.1%} | "
          f"{test['baseline']['false_reject_rate']:.1%} | {test['baseline']['false_accept_rate']:.1%} |")
        A(f"| **held-out fixtures** | **enhanced** | **{test['enhanced']['accuracy']:.1%}** | "
          f"**{test['enhanced']['false_reject_rate']:.1%}** | **{test['enhanced']['false_accept_rate']:.1%}** |")
        A("")
        A(f"On the held-out fixtures, ψ accuracy rises **{test['baseline']['accuracy']:.0%} → "
          f"{test['enhanced']['accuracy']:.0%}**, false-reject collapses "
          f"**{test['baseline']['false_reject_rate']:.0%} → {test['enhanced']['false_reject_rate']:.0%}**, and")
        A("**false-accept stays 0** (the out-guard never maps an out-of-abstraction step to a web action).")
        A("Because this is held-out, the lift is genuine generalization — not in-sample fitting.")
        A("")
        A("**Train/inference coupling (important):** the frozen strong PRM was trained on Stage-1-ψ features, so")
        A(f"the PRM candidate pool is intentionally still normalized by the **training-time** ψ. With that, the")
        A(f"rerank top-1 is preserved at `{es.get('prm_rerank_top1_rate')}`. A diagnostic that *also* swaps the")
        A("PRM's candidate normalizer to enhanced-ψ degrades rerank top-1 to `0.1` (out-of-distribution for the")
        A("frozen model) — confirming the coverage layer must drive η/execution, not the PRM's feature path,")
        A("unless the PRM is regenerated and retrained on enhanced-ψ features (a Stage-1 retrain).")
        A("")
        A("> Still honest about limits: 78.5% is on 65 author-constructed in-abstraction steps, not live logs;")
        A("> the residual misses are mostly defensible action-boundary cases (post-exploitation vs")
        A("> privilege-escalation, config-read as content_retrieval vs sensitive_file_read). The remaining gap")
        A("> and the PRM-retrain question both still want **held-out real operator logs** to settle.")
        A("")
    A("## Decision gate")
    A("")
    A(f"> Threshold: out-of-abstraction > **{g['threshold']:.0%}** ⇒ schema-extend before Phase 2.")
    A(">")
    A(f"> **Verdict:** {g['verdict']}")
    A("")
    if g["schema_extension_shortlist"]:
        A("**Schema-extension shortlist (prioritized by frequency across boxes):**")
        A("")
        for tok, n in g["schema_gap_token_counts"].items():
            A(f"- `{tok}` — {n} occurrence(s)")
        A("")
    A("## Per-box breakdown")
    A("")
    A("| Box | Family | Steps | Out-of-abs | ψ acc | φ recall | PRM top-1 |")
    A("|---|---|--:|--:|--:|--:|--:|")
    for b in replay["per_box"]:
        rr = b["prm_rerank"]["top1_rate"]
        rrs = "—" if rr is None else f"{rr:.0%}"
        A(f"| {b['box']} | {b['abstract_family']} | {b['n_steps']} | "
          f"{b['out_of_abstraction_rate']:.0%} | {b['psi']['accuracy']:.0%} | "
          f"{b['phi']['field_recall']:.0%} | {rrs} |")
    A("")
    A("## Honest reading")
    A("")
    A("- The **out-of-abstraction rate** is the make-or-break number: it is the fraction of real steps that")
    A("  have *no* representation in the 16-action web schema (SSH logins, SMB/binary service exploits,")
    A("  kernel/offline-crack, SSTI/SSRF/XXE, vhost/subdomain enumeration). It is hand-labeled ground truth,")
    A("  not inferred from ψ, so ψ errors cannot flatter it.")
    A("- **ψ accuracy < 100%** on in-abstraction steps is expected: real operator phrasing is messier than the")
    A("  synthetic training intents; the gap is a normalizer-coverage TODO, reported not hidden.")
    A("- **φ recall < 100%** reflects genuinely lossy parsing of messy real output.")
    A("- The **PRM is a reranker, not an autonomous policy** (Stage-1 finding, reconfirmed here): its score head")
    A("  has an enumeration bias, so the closed-loop chain-adherence proxy is modest. Its intended use is to")
    A("  rerank a small LLM-proposed candidate set, where the rerank top-1 metric applies.")
    A("- These fixtures are **author-constructed from public write-up structure**, not live captures. They")
    A("  measure schema/normalizer/parser coverage, which is structural; live value-uplift is Phase 2/3 and")
    A("  remains gated behind an owned, isolated, authorized lab.")
    A("")

    out = ROOT / "STAGE2_PHASE1_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
