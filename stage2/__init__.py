"""Stage 2 — real-lab adapter (φ/ψ/η) for WebAttackSim.

This package implements the *inference-stage* adapter described in STAGE2_PLAN.md.
It maps concrete real-target tool output (nmap/gobuster/sqlmap/...) onto the SAME
abstract `Observation` / `Action` schema the Stage-1 oracle + PRM were trained on, so
the abstract-trained Pentest-PRM can rerank real candidate actions.

Three maps:
  * φ (phi)  : real tool output  -> AbstractWebState (`Observation`)            [NEW]
  * ψ (psi)  : LLM action text   -> AbstractWebAction (Stage-1 normalizer)      [reused]
  * η (eta)  : abstract action   -> concrete command (sandboxed, GATED)         [NEW]

SAFETY: η performs NO live execution by default. `stage2.eta.LiveExecutor` refuses to
run unless an explicit authorization + isolated-environment flag is set. Phase-1 here is
*offline replay only*: it parses recorded tool output and measures the abstraction gap.
NOTHING in this package contacts a network target.
"""

from .phi import Phi, AbstractStateAccumulator  # noqa: F401

__all__ = ["Phi", "AbstractStateAccumulator"]
