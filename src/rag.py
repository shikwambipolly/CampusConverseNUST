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
from src.semantic_layer import get_semantic_layer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "chunks", "corpus_chunks.json")


def _tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 inverted index"""
    clean = normalize_text(text).lower()
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", clean)


class HybridRAGRetriever:
    """3-Way Hybrid Retriever (BM25 + TF-IDF + Chroma Dense Embeddings) over 1,745 NUST policy chunks"""

    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.corpus_path = corpus_path
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_corpus_and_index()

    def _load_corpus_and_index(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus chunks not found at: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.chunk_id_to_idx = {
            c.get("chunk_id", str(i)): i for i, c in enumerate(self.chunks)
        }
        self.prog_to_indices: Dict[str, List[int]] = {}
        for i, c in enumerate(self.chunks):
            first_line = c.get("text", "").split("\n", 1)[0].strip()
            if first_line.startswith("Programme:"):
                self.prog_to_indices.setdefault(first_line, []).append(i)

        # 1. Build BM25 Index
        corpus_tokens = [_tokenize(c["text"] + " " + c.get("section_title", "")) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

        # 2. Build TF-IDF Semantic Index
        corpus_texts = [f"{c.get('section_title', '')} {c['text']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=20000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Executes 3-way hybrid search (BM25 + TF-IDF + Chroma dense embeddings),
        merging ranks via Reciprocal Rank Fusion (RRF).
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        pool_size = max(top_k * 20, 120)

        # 1. BM25 scoring & ranking
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:pool_size]

        # 2. TF-IDF Cosine scoring & ranking
        q_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        tfidf_ranked_indices = np.argsort(tfidf_scores)[::-1][:pool_size]

        # 3. Dense Chroma Vector Search (if available)
        chroma_ranked_indices: List[int] = []
        chroma_distances: Dict[int, float] = {}
        try:
            semantic = get_semantic_layer()
            dense_hits = semantic.search_policies_and_regulations(query=query, n_results=min(pool_size, 60))
            for hit in dense_hits:
                cid = hit.get("chunk_id")
                if cid and cid in self.chunk_id_to_idx:
                    idx = self.chunk_id_to_idx[cid]
                    chroma_ranked_indices.append(idx)
                    chroma_distances[idx] = hit.get("relevance_distance", 1.0)
        except Exception:
            pass

        # 4. Reciprocal Rank Fusion (RRF) across all active rankers
        rrf_scores: Dict[int, float] = {}

        for rank, idx in enumerate(bm25_ranked_indices):
            if bm25_scores[idx] > 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(tfidf_ranked_indices):
            if tfidf_scores[idx] > 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(chroma_ranked_indices):
            # Weight dense semantic vector rank strongly in RRF
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.25 / (rrf_k + rank + 1))

        if not rrf_scores:
            return []

        # 5. Exact Course Code Boosting (when query mentions a specific NUST module code like DSA612S)
        q_codes = [m.lower() for m in re.findall(r"\b[A-Za-z]{3}[0-9]{3}[A-Za-z0-9]\b", query)]
        if q_codes:
            for idx, chunk in enumerate(self.chunks):
                c_lower = chunk.get("text", "").lower()
                if any(code in c_lower for code in q_codes):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 0.045

        # 6. Curriculum table boosting for module/course breakdown queries
        lower_q = query.lower()
        is_curriculum_query = any(
            k in lower_q
            for k in ("module", "modules", "course", "courses", "curriculum", "subjects", "year 1", "year 2", "year 3", "year 4")
        )
        if is_curriculum_query:
            year_match = re.search(r"\byear\s*([1-4])\b", lower_q)
            target_year = f"year {year_match.group(1)}" if year_match else None

            # Propagate base relevance from top-matched Programme intro chunks to sibling curriculum table chunks
            prog_best_score: Dict[str, float] = {}
            for idx, score in rrf_scores.items():
                first_line = self.chunks[idx].get("text", "").split("\n", 1)[0].strip()
                if first_line.startswith("Programme:"):
                    if score > prog_best_score.get(first_line, 0.0):
                        prog_best_score[first_line] = score

            top_progs = sorted(prog_best_score.items(), key=lambda x: x[1], reverse=True)[:3]
            for prog_header, best_score in top_progs:
                for sib_idx in self.prog_to_indices.get(prog_header, []):
                    sib_body = self.chunks[sib_idx].get("text", "").split("\n", 1)[-1].lower()
                    if "semester " in sib_body or "curriculum" in sib_body:
                        rrf_scores[sib_idx] = max(rrf_scores.get(sib_idx, 0.0), best_score * 0.90)

            for idx in list(rrf_scores.keys()):
                c_text = self.chunks[idx].get("text", "")
                body_lower = c_text.split("\n", 1)[-1].lower()
                code_count = len(re.findall(r"\b[A-Z]{3}[0-9]{3}[A-Z0-9]\b", c_text))
                if code_count >= 3 and ("curriculum" in body_lower or "semester " in body_lower):
                    boost = 1.85 if (target_year and target_year in body_lower) else 1.45
                    rrf_scores[idx] *= boost
                    # If curriculum table spans _C1 and _C2 on the same page, ensure companion chunk is included
                    cid = self.chunks[idx].get("chunk_id", "")
                    if cid.endswith("_C1"):
                        comp_idx = self.chunk_id_to_idx.get(cid[:-3] + "_C2")
                        if comp_idx is not None:
                            rrf_scores[comp_idx] = max(rrf_scores.get(comp_idx, 0.0), rrf_scores[idx] * 0.95)

        # Sort by final fused RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            chunk = dict(self.chunks[idx])
            chunk["rrf_score"] = round(rrf_scores[idx], 5)
            chunk["bm25_score"] = round(float(bm25_scores[idx]), 3)
            chunk["tfidf_score"] = round(float(tfidf_scores[idx]), 3)
            if idx in chroma_distances:
                chunk["chroma_distance"] = chroma_distances[idx]
            results.append(chunk)

        return results

    def answer_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves top policy passages and formats them with strict citations.
        Includes `is_relevant` to distinguish genuine institutional policy matches
        from out-of-scope queries.
        """
        retrieved = self.retrieve(query, top_k=top_k)

        if not retrieved:
            return {
                "query": query,
                "found": False,
                "is_relevant": False,
                "chunks": [],
                "formatted_response": "No relevant NUST policy clauses were found for your query."
            }

        max_bm25 = max((c.get("bm25_score", 0.0) for c in retrieved), default=0.0)
        max_tfidf = max((c.get("tfidf_score", 0.0) for c in retrieved), default=0.0)
        min_chroma = min((c.get("chroma_distance", 2.0) for c in retrieved), default=2.0)

        # A query is relevant to NUST policies if lexical or dense similarity exceeds noise floor
        is_relevant = (max_bm25 >= 5.5) or (max_tfidf >= 0.16) or (min_chroma < 0.55)

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
            "is_relevant": is_relevant,
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
