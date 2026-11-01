import json
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Agents.State import ResearchState  # noqa: E402

RERANK_TOP_K = 5


def _rerank_cases(query: str, retrieved_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from Common.llm import get_llm

    class RankedCase(BaseModel):
        source_id: str = Field(
            min_length=1,
            description="Identifiant source exact d'une affaire candidate",
        )
        reason: str = Field(
            min_length=1,
            description="Justification concise de sa pertinence",
        )

    class RerankingResult(BaseModel):
        cases: list[RankedCase] = Field(
            description="Affaires candidates classees par pertinence decroissante"
        )

    parser = PydanticOutputParser(pydantic_object=RerankingResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu classes des affaires de jurisprudence pour la requete de l'utilisateur. "
                "Base-toi uniquement sur les informations des candidates. "
                "Retourne exactement {expected_count} affaires, de la plus pertinente a la "
                "moins pertinente. "
                "N'invente aucun source_id et n'en retourne aucun qui ne figure pas parmi les "
                "candidates. Chaque source_id doit apparaitre une seule fois. "
                "Fournis une justification courte pour chaque affaire.\n\n{format_instructions}",
            ),
            (
                "human",
                "Requete utilisateur :\n{query}\n\nAffaires candidates (JSON) :\n{candidates}",
            ),
        ]
    )
    chain = prompt | get_llm(temperature=0) | parser

    candidates_for_prompt = []
    cases_by_source_id = {}
    for candidate in retrieved_cases:
        metadata = candidate.get("metadatas") or {}
        source_id = candidate.get("source_id") or metadata.get("source_id")
        if source_id is None or not str(source_id).strip():
            raise ValueError("Every retrieved case must contain a source_id")
        source_id = str(source_id)
        if source_id in cases_by_source_id:
            raise ValueError(f"Duplicate retrieved source_id: {source_id}")

        summary = metadata.get("Résumé") or candidate.get("document") or ""
        cases_by_source_id[source_id] = (candidate, str(summary))
        candidates_for_prompt.append(
            {
                "source_id": source_id,
                "summary": str(summary),
                "juridiction": metadata.get("juridiction", ""),
                "themes": metadata.get("themes", ""),
                "date_decision": metadata.get("Date de decision", ""),
                "location": metadata.get("location", ""),
                "numero": metadata.get("Numéros", ""),
            }
        )

    expected_count = min(RERANK_TOP_K, len(cases_by_source_id))
    response = chain.invoke(
        {
            "query": query,
            "candidates": json.dumps(candidates_for_prompt, ensure_ascii=False),
            "format_instructions": parser.get_format_instructions(),
            "expected_count": expected_count,
        }
    )
    if len(response.cases) != expected_count:
        raise ValueError(
            f"LLM reranker returned {len(response.cases)} cases; "
            f"expected exactly {expected_count}"
        )

    selected_cases = []
    seen_source_ids = set()
    for rank, ranked_case in enumerate(response.cases, start=1):
        source_id = ranked_case.source_id
        if source_id not in cases_by_source_id:
            raise ValueError(f"LLM reranker returned unknown source_id: {source_id}")
        if source_id in seen_source_ids:
            raise ValueError(f"LLM reranker returned duplicate source_id: {source_id}")
        seen_source_ids.add(source_id)

        candidate, summary = cases_by_source_id[source_id]
        selected_cases.append(
            {
                **candidate,
                "source_id": source_id,
                "summary": summary,
                "rerank_rank": rank,
                "rerank_reason": ranked_case.reason,
            }
        )

    return selected_cases


def retrieval_agent(state: ResearchState) -> dict[str, Any]:
    """Re-rank retrieved cases and keep the five best candidates in the shared state."""
    query = state.get("query")
    if query is None or not query.strip():
        raise ValueError("ResearchState must contain a non-empty query")

    retrieved_cases = state.get("retrieved_cases", [])
    if not retrieved_cases:
        return {
            "selected_cases": [],
            "reranking": {
                "candidate_count": 0,
                "selected_count": 0,
                "selected_source_ids": [],
                "top_k": RERANK_TOP_K,
            },
        }

    selected_cases = _rerank_cases(query, retrieved_cases)
    return {
        "selected_cases": selected_cases,
        "reranking": {
            "candidate_count": len(retrieved_cases),
            "selected_count": len(selected_cases),
            "selected_source_ids": [
                case["source_id"] for case in selected_cases
            ],
            "top_k": RERANK_TOP_K,
        },
    }