import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.tools import tool

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chromadb  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from Agents.State import ResearchState, RetrievalMethod  # noqa: E402
from Retreival.HybridSearch import hybrid_search  # noqa: E402
from Retreival.KeywordSearch import KeywordSearch_BM25, KeywordSearch_TFIDF, tokenizer  # noqa: E402
from Retreival.SemanticSearch import semanticSearch  # noqa: E402

COLLECTION_NAME = "jurisprudence_chunk_300_overlap_140"
DEFAULT_METHOD: RetrievalMethod = "Hybrid"
DEFAULT_N_RESULTS = 100
DEFAULT_TOP_K = 10
RRF_K = 60
FALLBACK_ORDER: tuple[RetrievalMethod, ...] = ("BM25", "Semantic", "TF-IDF")

_index_lock = Lock()


def _load_all_documents(collection, batch_size: int = 500) -> dict[str, list]:
    documents = []
    ids = []
    metadatas = []
    offset = 0

    while True:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        batch_ids = batch["ids"]
        batch_documents = batch["documents"] or []
        batch_metadatas = batch["metadatas"] or []

        if not batch_ids:
            break
        if not (len(batch_ids) == len(batch_documents) == len(batch_metadatas)):
            raise ValueError("Chroma returned misaligned documents, IDs, and metadata")

        ids.extend(batch_ids)
        documents.extend(batch_documents)
        metadatas.extend(batch_metadatas)
        offset += len(batch_ids)

    if not documents:
        raise ValueError(f"Chroma collection {COLLECTION_NAME!r} contains no chunks")

    return {"documents": documents, "ids": ids, "metadatas": metadatas}


@lru_cache(maxsize=1)
def _get_search_context():
    client = chromadb.PersistentClient(path=str(BACKEND_DIR / "Collections"))
    collection = client.get_collection(name=COLLECTION_NAME)
    index = _load_all_documents(collection)
    return client, collection, index


def _prepare_keyword_index(method: RetrievalMethod, index: dict[str, Any]) -> None:
    with _index_lock:
        if method in ("TF-IDF", "Hybrid") and "tfidf_vectors" not in index:
            vectorizer = TfidfVectorizer()
            index["tfidf_vectorizer"] = vectorizer
            index["tfidf_vectors"] = vectorizer.fit_transform(index["documents"])

        if method == "BM25" and "bm25_index" not in index:
            tokenized_documents = tokenizer(
                index["documents"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            index["bm25_tokens"] = tokenized_documents
            index["bm25_index"] = BM25Okapi(tokenized_documents, k1=1.25)


def _retrieve(
    query: str,
    method: RetrievalMethod,
    n_results: int,
    top_k: int,
    collection,
    index: dict[str, Any],
) -> list[dict[str, Any]]:
    if method in ("TF-IDF", "BM25", "Hybrid"):
        _prepare_keyword_index(method, index)

    if method == "TF-IDF":
        return KeywordSearch_TFIDF(query=query, top_k=n_results, index=index)[:top_k]
    if method == "BM25":
        return KeywordSearch_BM25(query=query, top_k=n_results, index=index)[:top_k]
    if method == "Semantic":
        return semanticSearch(
            collection=collection,
            index=index,
            query=query,
            n_results=n_results,
            top_k=top_k,
        )
    if method == "Hybrid":
        return hybrid_search(
            query=query,
            collection=collection,
            index=index,
            keyword_search=KeywordSearch_TFIDF,
            n_results=n_results,
            top_k=top_k,
            rrf_k=RRF_K,
        )
    raise ValueError(f"Unsupported retrieval method: {method}")


def _search_cases(
    query: str,
    method: RetrievalMethod = DEFAULT_METHOD,
    n_results: int = DEFAULT_N_RESULTS,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if n_results <= 0:
        raise ValueError("n_results must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if method not in ("TF-IDF", "BM25", "Semantic", "Hybrid"):
        raise ValueError(f"Unsupported retrieval method: {method}")

    _, collection, index = _get_search_context()
    methods_to_try = (method,) + tuple(
        candidate for candidate in FALLBACK_ORDER if candidate != method
    )
    attempted_methods = []

    for candidate_method in methods_to_try:
        attempted_methods.append(candidate_method)
        cases = _retrieve(
            query=query,
            method=candidate_method,
            n_results=n_results,
            top_k=top_k,
            collection=collection,
            index=index,
        )
        if cases:
            return {
                "query": query,
                "collection": COLLECTION_NAME,
                "requested_method": method,
                "method_used": candidate_method,
                "fallback_used": candidate_method != method,
                "attempted_methods": attempted_methods,
                "n_results": n_results,
                "top_k": top_k,
                "cases": cases,
            }

    return {
        "query": query,
        "collection": COLLECTION_NAME,
        "requested_method": method,
        "method_used": None,
        "fallback_used": len(attempted_methods) > 1,
        "attempted_methods": attempted_methods,
        "n_results": n_results,
        "top_k": top_k,
        "cases": [],
    }


@tool
def search_cases(
    query: str,
    method: RetrievalMethod = DEFAULT_METHOD,
    n_results: int = DEFAULT_N_RESULTS,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Search jurisprudence cases and fall back when a method finds no cases."""
    return _search_cases(
        query=query,
        method=method,
        n_results=n_results,
        top_k=top_k,
    )


def research_agent(state: ResearchState) -> dict[str, Any]:
    """Run case retrieval and save the selected cases and retrieval details in state."""
    query = state.get("query")
    if query is None:
        raise ValueError("ResearchState must contain a query")

    result = _search_cases(
        query=query,
        method=state.get("selected_method", DEFAULT_METHOD),
        n_results=state.get("n_results", DEFAULT_N_RESULTS),
        top_k=state.get("top_k", DEFAULT_TOP_K),
    )
    return {
        "retrieved_cases": result["cases"],
        "research": result,
    }
