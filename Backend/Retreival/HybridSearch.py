from collections.abc import Callable, Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[dict]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    fused: dict[str, dict] = {}
    for method_name, results in rankings.items():
        seen_source_ids: set[str] = set()
        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadatas") or {}
            source_id = result.get("source_id") or metadata.get("source_id")
            if not source_id:
                raise ValueError(f"{method_name} result at rank {rank} has no source_id")
            if source_id in seen_source_ids:
                raise ValueError(f"{method_name} returned duplicate source_id {source_id}")
            seen_source_ids.add(source_id)

            if source_id not in fused:
                fused[source_id] = {
                    "source_id": source_id,
                    "rrf_score": 0.0,
                    "ranks": {},
                    "document": result.get("document", ""),
                    "id": result.get("id"),
                    "metadatas": metadata,
                    "_best_rank": rank,
                }

            case = fused[source_id]
            case["rrf_score"] += 1 / (rrf_k + rank)
            case["ranks"][method_name] = rank
            if rank < case["_best_rank"]:
                case["document"] = result.get("document", "")
                case["id"] = result.get("id")
                case["metadatas"] = metadata
                case["_best_rank"] = rank

    ranked_cases = sorted(
        fused.values(),
        key=lambda case: (-case["rrf_score"], case["source_id"]),
    )
    for case in ranked_cases:
        del case["_best_rank"]
    return ranked_cases[:top_k]


def hybrid_search(
    query: str,
    collection,
    index: dict,
    keyword_search: Callable,
    n_results: int = 100,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    if n_results <= 0:
        raise ValueError("n_results must be positive")

    keyword_results = keyword_search(
        query=query,
        top_k=n_results,
        index=index,
    )

    from Retreival.SemanticSearch import semanticSearch

    semantic_results = semanticSearch(
        collection=collection,
        index=index,
        query=query,
        n_results=n_results,
        top_k=n_results,
    )

    return reciprocal_rank_fusion(
        {
            "keyword": keyword_results,
            "semantic": semantic_results,
        },
        top_k=top_k,
        rrf_k=rrf_k,
    )
