from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class WorkflowState(TypedDict, total=False):
    feature_name: str
    feature_description: str

    expanded_query: str
    candidate_domains: List[str]
    candidate_laws: List[str]
    region_hints: List[str]
    routing_confidence: float

    retrieved_evidence: List[Dict[str, Any]]

    needs_geo_compliance: bool
    reasoning: str
    citations: List[str]
    deterministic_confidence: float
    llm_confidence: float
    classification_conflict: bool
    classification_confidence: float
    needs_hitl: bool
    hitl_reason: str

    audit_trail: List[str]
    output: Dict[str, Any]
