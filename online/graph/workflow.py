from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from offline.qdrant_config import QdrantSettings
from offline.retriever import ComplianceRetriever
from online.graph.nodes import classify, finalize, hitl_router, query_enhance_route
from online.graph.nodes.retrieve import make_node as make_retrieve_node
from online.graph.state import WorkflowState


def build_app(settings: QdrantSettings | None = None):
    retriever = ComplianceRetriever(settings=settings)

    graph = StateGraph(WorkflowState)
    graph.add_node("query_enhancer", query_enhance_route.make_node(retriever.available_domains))
    graph.add_node("retrieve", make_retrieve_node(retriever.search, retriever.available_domains))
    graph.add_node("classifier", classify.run)
    graph.add_node("hitl_router", hitl_router.run)
    graph.add_node("finalize", finalize.run)

    graph.add_edge(START, "query_enhancer")
    graph.add_edge("query_enhancer", "retrieve")
    graph.add_edge("retrieve", "classifier")
    graph.add_edge("classifier", "hitl_router")
    graph.add_edge("hitl_router", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
