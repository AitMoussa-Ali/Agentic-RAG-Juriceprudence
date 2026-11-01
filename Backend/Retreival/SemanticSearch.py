from math import log2
from statistics import mean

def _rank_cases(scores, candidate_indices, index, top_k):
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    best_chunks = {}
    for chunk_index in candidate_indices:
        metadata = index["metadatas"][chunk_index]
        source_id = metadata.get("source_id")
        if not source_id:
            raise ValueError(f"Chunk {index['ids'][chunk_index]} has no source_id metadata")

        score = float(scores[chunk_index])
        current = best_chunks.get(source_id)
        if current is None or score > current["score"]:
            best_chunks[source_id] = {
                "document": index["documents"][chunk_index],
                "score": score,
                "id": index["ids"][chunk_index],
                "source_id": source_id,
                "metadatas": metadata,
            }

    ranked_cases = sorted(
        best_chunks.values(),
        key=lambda result: (result["score"], result["source_id"]),
    )
    return ranked_cases[:top_k]

def semanticSearch( collection, index: dict, query : str, n_results : int = 50, top_k : int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    
    scores = results["distances"][0]
    ids = results["ids"][0]
    candidate_indices = []
    scores_dict = [0]*len(index["ids"])
    for id in ids:
        i = index["ids"].index(id)
        candidate_indices.append(i)
        scores_dict[i] = scores[ids.index(id)]
    ranked_cases = _rank_cases(scores_dict, candidate_indices, index, top_k)
    return ranked_cases

def evaluate_strategy(
    n_results: list,
    collection,
    search_function,
    method_name,
    evaluation_queries,
    index,
    K_VALUES=(2, 5, 10),
):
    per_query = []
    max_k = max(K_VALUES)

    for query_data in evaluation_queries:
        relevant_ids = set(query_data["relevant_source_ids"])

        if not relevant_ids:
            raise ValueError(
                f"{query_data['query_id']} has no relevance labels"
            )

        for n in n_results:
            ranked_results = search_function(
                collection=collection,
                query=query_data["query"],
                top_k=max_k,
                index=index,
                n_results=n,
            )

            ranked_ids = [
                result["source_id"] for result in ranked_results
            ]

            if len(ranked_ids) != len(set(ranked_ids)):
                raise ValueError(
                    f"{method_name} returned duplicate cases"
                )

            row = {
                "method": method_name,
                "query_id": query_data["query_id"],
                "query": query_data["query"],
                "n_results": n,
                "retrieved_source_ids": ranked_ids,
            }

            for k in K_VALUES:
                top_k_ids = ranked_ids[:k]

                found = sum(
                    source_id in relevant_ids
                    for source_id in top_k_ids
                )

                row[f"precision@{k}"] = found / k
                row[f"recall@{k}"] = found / len(relevant_ids)

                dcg = sum(
                    1 / log2(rank + 1)
                    for rank, source_id in enumerate(
                        top_k_ids, start=1
                    )
                    if source_id in relevant_ids
                )

                ideal_count = min(len(relevant_ids), k)

                ideal_dcg = sum(
                    1 / log2(rank + 1)
                    for rank in range(1, ideal_count + 1)
                )

                row[f"ndcg@{k}"] = (
                    dcg / ideal_dcg if ideal_dcg else 0.0
                )

            per_query.append(row)

    summary = {
        "method": method_name,
        "queries": len(evaluation_queries),
        "evaluations": len(per_query),
    }

    for n in n_results:
        rows_for_n = [
            row for row in per_query
            if row["n_results"] == n
        ]

        for k in K_VALUES:
            for metric in ("precision", "recall", "ndcg"):
                metric_name = f"{metric}@{k}"

                summary[f"{metric_name}_{n}"] = mean(
                    row[metric_name] for row in rows_for_n
                )

    return summary, per_query

