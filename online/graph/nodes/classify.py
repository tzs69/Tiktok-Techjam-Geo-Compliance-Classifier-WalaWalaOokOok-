from __future__ import annotations

import os
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from online.graph.state import WorkflowState

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None


class ClassificationLLMOutput(BaseModel):
    needs_geo_compliance: bool = Field(default=False)
    reasoning: str = Field(default="")
    confidence: float = Field(default=0.5)


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _deterministic_decision(
    evidence: List[Dict[str, object]],
    candidate_laws: List[str],
) -> Tuple[bool, float]:
    top_score = max((float(item.get("base_score", 0.0)) for item in evidence), default=0.0)
    top3 = sorted((float(item.get("base_score", 0.0)) for item in evidence), reverse=True)[:3]
    avg_top3 = sum(top3) / len(top3) if top3 else 0.0

    explicit_law_signal = 1.0 if candidate_laws else 0.0

    confidence = _clamp(
        0.50 * top_score
        + 0.35 * avg_top3
        + 0.15 * explicit_law_signal
    )

    needs_geo = bool(confidence >= 0.58)
    if explicit_law_signal and len(evidence) >= 2:
        needs_geo = True

    return needs_geo, confidence


def _llm_classify(
    query: str,
    evidence: List[Dict[str, object]],
    candidate_domains: List[str],
    candidate_laws: List[str],
    region_hints: List[str],
) -> ClassificationLLMOutput | None:
    if ChatOpenAI is None or not os.environ.get("OPENAI_API_KEY"):
        return None

    evidence_rows = []
    for item in evidence[:8]:
        evidence_rows.append(
            {
                "chunk_id": item.get("chunk_id", ""),
                "domain": item.get("domain", ""),
                "law_name": item.get("law_name", ""),
                "score": item.get("base_score", 0.0),
                "text": str(item.get("text", ""))[:600],
            }
        )

    prompt = (
        "You are a geo-compliance classifier. Use only provided evidence. "
        "Decide whether this feature needs geo-specific compliance logic. "
        "Return structured output with needs_geo_compliance (bool), reasoning (max 2 sentences), confidence (0..1)."
        f"\n\nQuery: {query}"
        f"\nCandidate domains: {candidate_domains}"
        f"\nCandidate laws: {candidate_laws}"
        f"\nRegion hints: {region_hints}"
        f"\nEvidence: {evidence_rows}"
    )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured = llm.with_structured_output(ClassificationLLMOutput)
        response = structured.invoke(prompt)
        response.confidence = _clamp(float(response.confidence))
        return response
    except Exception:
        return None


def run(state: WorkflowState) -> WorkflowState:
    query = state.get("expanded_query", "")
    evidence = list(state.get("retrieved_evidence", []))
    candidate_laws = list(state.get("candidate_laws", []))
    region_hints = list(state.get("region_hints", []))

    det_bool, det_conf = _deterministic_decision(evidence, candidate_laws)

    llm_output = _llm_classify(query, evidence, list(state.get("candidate_domains", [])), candidate_laws, region_hints)
    llm_used = llm_output is not None

    if llm_output:
        final_conf = _clamp(0.6 * det_conf + 0.4 * llm_output.confidence)
        if llm_output.confidence >= 0.55:
            final_bool = bool(llm_output.needs_geo_compliance)
        else:
            final_bool = det_bool
        reasoning = llm_output.reasoning.strip() or "Evidence was evaluated but reasoning was not provided by the model."
        conflict = final_bool != det_bool
    else:
        final_conf = det_conf
        final_bool = det_bool
        conflict = False
        if not evidence:
            reasoning = "No relevant legal evidence was retrieved from the knowledge base."
        elif final_bool:
            reasoning = "Retrieved evidence indicates potential jurisdiction-specific legal obligations for this feature."
        else:
            reasoning = "Retrieved evidence does not strongly indicate jurisdiction-specific legal obligations for this feature."

    citations: List[str] = []
    for item in evidence:
        citation = f"{item.get('law_name')} ({item.get('source_path')})"
        if citation not in citations:
            citations.append(citation)

    audit = list(state.get("audit_trail", []))
    audit.append(
        "classify: "
        f"decision={final_bool} confidence={round(final_conf, 4)} det_conf={round(det_conf, 4)} llm_used={llm_used}"
    )

    return {
        "needs_geo_compliance": final_bool,
        "reasoning": reasoning,
        "citations": citations[:5],
        "deterministic_confidence": float(det_conf),
        "llm_confidence": float(llm_output.confidence) if llm_output else 0.0,
        "classification_conflict": bool(conflict),
        "classification_confidence": float(final_conf),
        "audit_trail": audit,
    }
