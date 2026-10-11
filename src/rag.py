"""
src/rag.py

Hybrid Retrieval-Augmented Generation (RAG) Subsystem.
Adheres strictly to spec.md Section 3.5:
- Dual-index retrieval: Sparse BM25 + Cosine TF-IDF
- Reciprocal Rank Fusion (RRF) rank aggregation:
  RRF_Score(d) = sum(1 / (k + rank_m(d))) for m in {bm25, tfidf}, k=60
- Top k=3 passage assembly with strict institutional source citations
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import normalize_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "chunks", "corpus_chunks.json")


def _tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 inverted index"""
    clean = normalize_text(text).lower()
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", clean)


class HybridRAGRetriever:
    """Hybrid BM25 + TF-IDF Vector Retriever over 1,745 NUST policy chunks"""

    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.corpus_path = corpus_path
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_corpus_and_index()

    def _load_corpus_and_index(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus chunks not found at: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # 1. Build BM25 Index
        corpus_tokens = [_tokenize(c["text"] + " " + c.get("section_title", "")) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

        # 2. Build TF-IDF Semantic Index
        corpus_texts = [f"{c.get('section_title', '')} {c['text']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=8000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Executes hybrid BM25 and TF-IDF search, merging ranks via Reciprocal Rank Fusion (RRF).
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # 1. BM25 scoring & ranking
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:top_k * 5]

        # 2. TF-IDF Cosine scoring & ranking
        q_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        tfidf_ranked_indices = np.argsort(tfidf_scores)[::-1][:top_k * 5]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}

        for rank, idx in enumerate(bm25_ranked_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(tfidf_ranked_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by final fused RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            chunk = dict(self.chunks[idx])
            chunk["rrf_score"] = round(rrf_scores[idx], 5)
            chunk["bm25_score"] = round(float(bm25_scores[idx]), 3)
            results.append(chunk)

        return results

    def answer_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves top policy passages and formats them with strict citations.
        """
        retrieved = self.retrieve(query, top_k=top_k)

        if not retrieved:
            return {
                "query": query,
                "found": False,
                "chunks": [],
                "formatted_response": "No relevant NUST policy clauses were found for your query."
            }

        citations = []
        for i, c in enumerate(retrieved, 1):
            src = c.get("source_document", "NUST Institutional Corpus")
            sec = c.get("section_title", "General Regulation")
            page = c.get("page_number", "N/A")
            excerpt = c["text"].strip().replace("\n", " ")
            citations.append(
                f"[{i}] {excerpt}\n"
                f"    [Source: {src} | Section: {sec} | Page: {page}]"
            )

        formatted = (
            f"Official NUST Policy Information:\n\n" +
            "\n\n".join(citations)
        )

        return {
            "query": query,
            "found": True,
            "chunks": retrieved,
            "formatted_response": formatted
        }


# Global singleton instance for efficient in-memory reuse
_retriever_instance: Optional[HybridRAGRetriever] = None


def get_rag_retriever() -> HybridRAGRetriever:
    """Returns singleton instance of HybridRAGRetriever"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRAGRetriever()
    return _retriever_instance


def search_policy(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Convenience search interface for routing subsystem"""
    retriever = get_rag_retriever()
    return retriever.answer_query(query, top_k=top_k)
