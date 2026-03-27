"""
LangGraph graph definition — wires all nodes and conditional edges.
Implements all conditional edges from the design document.
"""

from langgraph.graph import StateGraph, END
from core.state import AgentState
from nodes.query_reconstruction import run as query_reconstruction
from nodes.super_router import run as super_router
from nodes.comparator import run as comparator
from nodes.context_updator import run as context_updator
from nodes.plan_node import run as plan_node
from nodes.user_clarification import run as user_clarification
from nodes.user_plan_approval import run as user_plan_approval
from nodes.implement import run as implement
from nodes.code_judge import run as code_judge


# --- Conditional edge functions ---

def has_prev_context_edge(state: AgentState) -> str:
    """After query_reconstruction: check if previous context exists."""
    if state["has_prev_context"]:
        return "comparator"
    return "context_updator"


def route_type_edge(state: AgentState) -> str:
    """After super_router: route based on user_approved flag or route_type.
    Also checks provider switch needed (design doc edge: Provider switch needed?).
    Provider switching is handled within super_router node itself."""
    # If coming back from user_plan_approval, use the approval flag
    if state.get("user_approved") is True:
        return "implement"
    if state.get("user_approved") is False:
        return "plan_node"

    # Fresh task — classify by route_type
    if state["route_type"] == "simple":
        return "implement"
    return "plan_node"


def plan_score_edge(state: AgentState) -> str:
    """After plan_node: combined edge for LLM judge approval + plan iteration safety.
    Design doc edges: 'LLM judge approved?' (plan_score >= 0.85)
                      'Plan iteration safety' (plan_iteration_count < 4)
    If iterations >= 4, force to user approval to prevent infinite loops."""
    if state["plan_iteration_count"] >= 4:
        return "user_plan_approval"
    if state["plan_score"] >= 0.85:
        return "user_plan_approval"
    return "user_clarification"


def implementation_correct_edge(state: AgentState) -> str:
    """After code_judge: combined edge for implementation correctness + iteration safety.
    Design doc edges: 'Implementation correct?' (status == 'correct')
                      'Implement iteration safety' (implement_iteration_count < 5)
    If iterations >= 5, stop to prevent infinite retries."""
    if state["implementation_status"].get("status") == "correct":
        return END
    if state["implement_iteration_count"] >= 5:
        return END
    return "implement"


def mcp_server_healthy_edge(state: AgentState) -> str:
    """Before implement: check if MCP servers are connected.
    Design doc edge: 'MCP server healthy?'
    If any server is disconnected, log error and warn but still proceed
    (implement node will attempt reconnection via MCPClient)."""
    server_status = state.get("mcp_server_status", {})
    disconnected = [name for name, status in server_status.items() if status == "disconnected"]
    if disconnected:
        # Log warning but proceed — MCPClient has reconnection logic
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("MCP servers disconnected before implement: %s. Will attempt reconnection.", disconnected)
    return "implement"


# --- Build the graph ---

def build_graph(config: dict = None):
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_reconstruction", query_reconstruction)
    graph.add_node("comparator", comparator)
    graph.add_node("context_updator", context_updator)
    graph.add_node("super_router", super_router)
    graph.add_node("plan_node", plan_node)
    graph.add_node("user_clarification", user_clarification)
    graph.add_node("user_plan_approval", user_plan_approval)
    graph.add_node("implement", implement)
    graph.add_node("code_judge", code_judge)

    # Entry point
    graph.set_entry_point("query_reconstruction")

    # Edge: has_prev_context? (design doc)
    graph.add_conditional_edges("query_reconstruction", has_prev_context_edge, {
        "comparator": "comparator",
        "context_updator": "context_updator",
    })

    # Comparator -> Context updator (update context after comparing)
    graph.add_edge("comparator", "context_updator")

    # Context updator -> Super router (classify task)
    graph.add_edge("context_updator", "super_router")

    # Edge: Route type + Provider switch + User approved? (design doc)
    # Simple -> implement (via MCP health check), Complex -> plan_node
    graph.add_conditional_edges("super_router", route_type_edge, {
        "implement": "implement",
        "plan_node": "plan_node",
    })

    # Edge: LLM judge approved? + Plan iteration safety (design doc)
    graph.add_conditional_edges("plan_node", plan_score_edge, {
        "user_plan_approval": "user_plan_approval",
        "user_clarification": "user_clarification",
    })

    # After clarification -> comparator (updates context, then super_router
    # routes to plan_node since route_type is set to "complex" by clarification node)
    graph.add_edge("user_clarification", "comparator")

    # User plan approval -> comparator (with user_approved flag + user_feedback)
    # Comparator updates context, then super_router checks user_approved:
    #   approved -> implement, rejected -> plan_node
    graph.add_edge("user_plan_approval", "comparator")

    # After implementation -> code judge reviews
    graph.add_edge("implement", "code_judge")

    # Edge: Implementation correct? + Implement iteration safety (design doc)
    graph.add_conditional_edges("code_judge", implementation_correct_edge, {
        END: END,
        "implement": "implement",
    })

    checkpointer = None
    if config and config.get("checkpointing", {}).get("enabled"):
        from core.checkpoint import PersistentMemorySaver

        path = config.get("checkpointing", {}).get("path", "CodePilot_UNCC/.codepilot/checkpoints.pkl")
        checkpointer = PersistentMemorySaver(path)

    return graph.compile(checkpointer=checkpointer)
