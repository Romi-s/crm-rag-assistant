from langgraph.graph import END, StateGraph

from app.agent.nodes import format_response, generate, retrieve, validate_input
from app.agent.state import QAState


def _should_continue(state: QAState) -> str:
    return "abort" if state.get("error") else "continue"


def build_graph():
    builder = StateGraph(QAState)

    builder.add_node("validate", validate_input)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("format", format_response)

    builder.set_entry_point("validate")
    builder.add_conditional_edges("validate", _should_continue,
                                  {"continue": "retrieve", "abort": END})
    builder.add_conditional_edges("retrieve", _should_continue,
                                  {"continue": "generate", "abort": END})
    builder.add_conditional_edges("generate", _should_continue,
                                  {"continue": "format", "abort": END})
    builder.add_edge("format", END)

    return builder.compile()


qa_graph = build_graph()
