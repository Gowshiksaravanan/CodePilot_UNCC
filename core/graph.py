"""
LangGraph graph definition — wires all nodes and conditional edges.
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
    """After super_router: simple -> implement, complex -> plan_node."""
    if state["route_type"] == "simple":
        return "implement"
    return "plan_node"


def plan_score_edge(state: AgentState) -> str:
    """After plan_node: if LLM judge approves (>= 0.85), show to user. Else clarify."""
    if state["plan_iteration_count"] >= 4:
        return "user_plan_approval"
    if state["plan_score"] >= 0.85:
        return "user_plan_approval"
    return "user_clarification"


def implementation_correct_edge(state: AgentState) -> str:
    """After code_judge: correct -> done, incorrect -> retry implement."""
    if state["implementation_status"].get("status") == "correct":
        return END
    if state["implement_iteration_count"] >= 5:
        return END
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

    # Query reconstruction -> check if previous context exists
    graph.add_conditional_edges("query_reconstruction", has_prev_context_edge, {
        "comparator": "comparator",
        "context_updator": "context_updator",
    })

    # Comparator -> Context updator (update context after comparing)
    graph.add_edge("comparator", "context_updator")

    # Context updator -> Super router (classify task)
    graph.add_edge("context_updator", "super_router")

    # Super router: simple -> implement, complex -> plan_node
    graph.add_conditional_edges("super_router", route_type_edge, {
        "implement": "implement",
        "plan_node": "plan_node",
    })

    # Plan score: approved -> user_plan_approval, rejected -> user_clarification
    graph.add_conditional_edges("plan_node", plan_score_edge, {
        "user_plan_approval": "user_plan_approval",
        "user_clarification": "user_clarification",
    })

    # After clarification -> back to plan_node to regenerate
    graph.add_edge("user_clarification", "plan_node")

    # User plan approval always -> query_reconstruction
    # Approved: feedback tells super_router to go straight to implement
    # Rejected: feedback contains user changes, triggers re-planning
    graph.add_edge("user_plan_approval", "query_reconstruction")

    # After implementation -> code judge reviews
    graph.add_edge("implement", "code_judge")

    # Code judge: correct -> END, incorrect -> retry implement
    graph.add_conditional_edges("code_judge", implementation_correct_edge, {
        END: END,
        "implement": "implement",
    })

    return graph.compile()
