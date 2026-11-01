import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from Agents.QA_agent import qa_agent  # noqa: E402
from Agents.Research_agent import research_agent  # noqa: E402
from Agents.Retreival_agent import retrieval_agent  # noqa: E402
from Agents.State import ResearchState  # noqa: E402


def _route_request(state: ResearchState) -> dict[str, Any]:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from Common.llm import get_llm

    user_message = state.get("user_message")
    if user_message is None or not user_message.strip():
        raise ValueError("ResearchState must contain a non-empty user_message")

    class RoutingDecision(BaseModel):
        intent: Literal["search_cases", "ask_question"] = Field(
            description=(
                "search_cases pour trouver des affaires sur un sujet; "
                "ask_question pour répondre à une question sur les affaires "
                "déjà sélectionnées dans la session"
            )
        )

    parser = PydanticOutputParser(pydantic_object=RoutingDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es le routeur d'un système de recherche en jurisprudence. "
                "Choisis une seule intention :\n"
                "- search_cases si le message demande de trouver, rechercher ou lister "
                "des affaires sur un sujet.\n"
                "- ask_question si le message pose une question sur les affaires déjà "
                "sélectionnées dans la session.\n"
                "Si aucune affaire n'est sélectionnée, choisis search_cases. "
                "N'explique pas ton raisonnement; retourne uniquement la décision structurée.\n\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Affaires déjà sélectionnées : {has_selected_cases}\n"
                "Message de l'utilisateur : {user_message}",
            ),
        ]
    )
    chain = prompt | get_llm(temperature=0) | parser
    decision = chain.invoke(
        {
            "user_message": user_message,
            "has_selected_cases": bool(state.get("selected_cases")),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    if decision.intent == "ask_question" and state.get("selected_cases"):
        return {
            "route": "qa",
            "user_question": user_message,
            "workflow_status": "answering_question",
        }

    return {
        "route": "research",
        "query": user_message,
        "user_question": "",
        "retrieved_cases": [],
        "research": {},
        "selected_cases": [],
        "reranking": {},
        "qa_answer": "",
        "qa_sources": [],
        "workflow_status": "searching_cases",
    }


def _run_research(state: ResearchState) -> dict[str, Any]:
    result = research_agent(state)
    return {
        **result,
        "workflow_status": (
            "cases_found" if result.get("retrieved_cases") else "no_cases_found"
        ),
    }


def _run_retrieval(state: ResearchState) -> dict[str, Any]:
    result = retrieval_agent(state)
    return {
        **result,
        "workflow_status": "cases_selected",
    }


def _run_qa(state: ResearchState) -> dict[str, Any]:
    result = qa_agent(state)
    return {
        **result,
        "workflow_status": "question_answered",
    }


def _route_after_monitoring(state: ResearchState) -> str:
    route = state.get("route")
    if route not in ("research", "qa"):
        raise ValueError(f"Monitoring selected an unsupported route: {route}")
    return route


def _route_after_research(state: ResearchState) -> str:
    return "retrieval" if state.get("retrieved_cases") else "end"


@lru_cache(maxsize=1)
def get_monitoring_graph():
    """Build and cache the LangGraph workflow for the research and QA agents."""
    graph = StateGraph(ResearchState)
    graph.add_node("monitoring", _route_request)
    graph.add_node("research", _run_research)
    graph.add_node("retrieval", _run_retrieval)
    graph.add_node("qa", _run_qa)

    graph.add_edge(START, "monitoring")
    graph.add_conditional_edges(
        "monitoring",
        _route_after_monitoring,
        {
            "research": "research",
            "qa": "qa",
        },
    )
    graph.add_conditional_edges(
        "research",
        _route_after_research,
        {
            "retrieval": "retrieval",
            "end": END,
        },
    )
    graph.add_edge("retrieval", END)
    graph.add_edge("qa", END)
    return graph.compile()


def monitoring_agent(state: ResearchState) -> dict[str, Any]:
    """Route a user message through the compiled multi-agent LangGraph workflow."""
    return get_monitoring_graph().invoke(state)
