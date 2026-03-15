"""
Shared state schema for the LangGraph agent.
All nodes read from and write to this state.
"""

from typing import TypedDict, List


class AgentState(TypedDict):
    # User input
    user_query: str
    restructured_query: str

    # Codebase context
    context: dict
    code_base: dict
    context_difference: str
    has_prev_context: bool

    # Routing
    route_type: str  # "simple" or "complex"

    # Planning
    current_plan: str
    plan_score: float
    plan_feedback: str
    plan_iteration_count: int
    plan_history: List[str]

    # User clarification
    clarification_questions: List[str]
    suggestions: List[List[str]]
    user_responses: List[str]
    user_approved: bool
    user_feedback: str

    # Implementation
    implementation_status: dict
    files_modified: List[str]
    implement_iteration_count: int
    execution_log: List[dict]

    # Conversation
    messages: list

    # Session
    session_id: str
    time_started: str

    # Debugging
    current_node: str
    error_log: List[dict]
