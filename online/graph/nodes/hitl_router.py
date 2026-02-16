from __future__ import annotations

from typing import List

from online.graph.state import WorkflowState


def run(state: WorkflowState) -> WorkflowState:
    evidence = list(state.get("retrieved_evidence", []))
    det_conf = float(state.get("deterministic_confidence", 0.0))
    llm_conf = float(state.get("llm_confidence", 0.0))
    final_conf = float(state.get("classification_confidence", 0.0))
    conflict = bool(state.get("classification_conflict", False))

    reasons: List[str] = []

    # Deterministic-first checks.
    if len(evidence) < 2:
        reasons.append("Insufficient evidence chunks for high-confidence decision.")
    if det_conf < 0.40:
        reasons.append("Low deterministic confidence (<0.40).")
    elif det_conf < 0.58:
        reasons.append("Deterministic confidence is in review band (0.40-0.58).")
    if conflict:
        reasons.append("Deterministic and LLM decisions conflict.")

    # Secondary LLM signal checks.
    if llm_conf > 0.0 and abs(llm_conf - det_conf) >= 0.30:
        reasons.append("Large gap between deterministic and LLM confidence.")
    elif 0.40 <= final_conf < 0.58 and not reasons:
        reasons.append("Blended confidence is in review band (0.40-0.58).")

    needs_hitl = bool(reasons)
    hitl_reason = reasons[0] if reasons else ""

    audit = list(state.get("audit_trail", []))
    audit.append(
        "hitl_router: "
        f"needs_hitl={needs_hitl} det_conf={round(det_conf, 4)} llm_conf={round(llm_conf, 4)} final_conf={round(final_conf, 4)}"
    )

    return {
        "needs_hitl": needs_hitl,
        "hitl_reason": hitl_reason,
        "audit_trail": audit,
    }
