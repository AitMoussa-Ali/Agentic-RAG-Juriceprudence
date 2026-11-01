from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
from rank_bm25 import BM25Okapi
import chromadb
from math import log2
from statistics import mean

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

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
        key=lambda result: (-result["score"], result["source_id"]),
    )
    return ranked_cases[:top_k]

#TF-IDF search function
def KeywordSearch_TFIDF(query: str, top_k: int, index: dict) -> list:
    query_vector = index["tfidf_vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["tfidf_vectors"]).ravel()
    candidate_indices = [i for i, score in enumerate(scores) if score > 0]
    return _rank_cases(scores, candidate_indices, index, top_k)

#BM25 search function
def KeywordSearch_BM25(query: str, top_k: int, index: dict) -> list:
    query_tokens = tokenizer(query, add_special_tokens=False)["input_ids"]
    if not query_tokens:
        return []

    scores = index["bm25_index"].get_scores(query_tokens)
    unique_query_tokens = set(query_tokens)
    candidate_indices = [
        i for i, document_tokens in enumerate(index["bm25_tokens"])
        if unique_query_tokens.intersection(document_tokens)
    ]
    return _rank_cases(scores, candidate_indices, index, top_k)


def evaluate_strategy(search_function, method_name, evaluation_queries, index, K_VALUES=(2, 5, 10)):
    per_query = []
    max_k = max(K_VALUES)

    for query_data in evaluation_queries:
        relevant_ids = set(query_data["relevant_source_ids"])
        if not relevant_ids:
            raise ValueError(f"{query_data['query_id']} has no relevance labels")

        ranked_results = search_function(
            query=query_data["query"],
            top_k=max_k,
            index=index,
        )
        ranked_ids = [result["source_id"] for result in ranked_results]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"{method_name} returned duplicate cases")

        row = {
            "method": method_name,
            "query_id": query_data["query_id"],
            "query": query_data["query"],
            "retrieved_source_ids": ranked_ids,
        }
        relevant_ranks = [
            rank for rank, source_id in enumerate(ranked_ids, start=1)
            if source_id in relevant_ids
        ]
        list = []
        for k in K_VALUES:
            found = sum(source_id in relevant_ids for source_id in ranked_ids[:k])
            row[f"precision@{k}"] = found / k
            row[f"recall@{k}"] = found / len(relevant_ids)
            list.append(f"precision@{k}")
            list.append(f"recall@{k}")
            list.append(f"ndcg@{k}")

            dcg = sum(
                1 / log2(rank + 1)
                for rank, source_id in enumerate(ranked_ids[:k], start=1)
                if source_id in relevant_ids
            )
            ideal_count = min(len(relevant_ids), k)
            ideal_dcg = sum(
                1 / log2(rank + 1)
                for rank in range(1, ideal_count + 1)
            )
            row[f"ndcg@{k}"] = dcg / ideal_dcg if ideal_dcg else 0.0

        per_query.append(row)

    summary = {"method": method_name, "queries": len(per_query)}
    for metric in list:
        summary[metric] = mean(row[metric] for row in per_query)
    return summary, per_query