from typing import Any, Literal, TypedDict

RetrievalMethod = Literal["TF-IDF", "BM25", "Semantic", "Hybrid"]
AgentRoute = Literal["research", "qa"]


class ResearchState(TypedDict, total=False):
    user_message: str
    route: AgentRoute
    workflow_status: str
    query: str
    selected_method: RetrievalMethod
    n_results: int
    top_k: int
    retrieved_cases: list[dict[str, Any]]
    research: dict[str, Any]
    selected_cases: list[dict[str, Any]]
    reranking: dict[str, Any]
    user_question: str
    qa_answer: str
    qa_sources: list[dict[str, Any]]
