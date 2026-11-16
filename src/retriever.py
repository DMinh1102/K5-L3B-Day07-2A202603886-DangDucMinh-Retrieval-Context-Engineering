"""Retrieval engine for Day 7 Agentic RAG.
Implements:
1. DenseRetriever (FastEmbed multilingual vector retrieval + Cosine Similarity)
2. BM25Retriever (Lexical BM25 with Vietnamese/Unicode normalization)
3. HybridRetriever (Reciprocal Rank Fusion - RRF combining Dense & BM25)
"""
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from fastembed import TextEmbedding


def normalize_text(text: str) -> list[str]:
    """Tokenize and normalize text using Unicode NFC."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class DenseRetriever:
    """Dense Vector Retriever using fastembed (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)."""
    def __init__(
        self,
        chunks_path: str | Path = "data/chunks.json",
        embeddings_path: str | Path = "data/embeddings.npy",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.chunks: list[dict[str, Any]] = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.chunk_by_id = {c["id"]: c for c in self.chunks}
        self.embeddings = np.load(embeddings_path)
        self.model = TextEmbedding(model_name=model_name)

    def _embed_query(self, query: str) -> np.ndarray:
        vecs = list(self.model.embed([query]))
        vec = np.array(vecs[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self._embed_query(query)
        # Cosine similarity via dot product (both vectors are normalized)
        scores = np.dot(self.embeddings, q_vec)

        # Apply metadata filters if provided
        results = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if filters:
                match = all(chunk.get("metadata", {}).get(k) == v for k, v in filters.items())
                if not match:
                    continue
            results.append({
                **chunk,
                "score": float(score),
                "strategy": "dense"
            })

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]


class BM25Retriever:
    """Lexical BM25 Retriever."""
    def __init__(self, chunks_path: str | Path = "data/chunks.json", k1: float = 1.5, b: float = 0.75):
        self.chunks: list[dict[str, Any]] = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.k1 = k1
        self.b = b
        self.corpus_size = len(self.chunks)
        self.doc_tokens = [normalize_text(c["text"]) for c in self.chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / (self.corpus_size or 1)

        # Compute document frequencies
        df = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1

        # Compute IDF
        self.idf = {
            term: math.log((self.corpus_size - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_terms = normalize_text(query)
        scores = []

        for idx, (chunk, tokens, doc_len) in enumerate(zip(self.chunks, self.doc_tokens, self.doc_lens)):
            if filters:
                match = all(chunk.get("metadata", {}).get(k) == v for k, v in filters.items())
                if not match:
                    continue

            term_counts = Counter(tokens)
            score = 0.0
            for term in q_terms:
                if term in self.idf:
                    tf = term_counts.get(term, 0)
                    if tf > 0:
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1)))
                        score += self.idf[term] * (numerator / denominator)

            if score > 0:
                scores.append({
                    **chunk,
                    "score": float(score),
                    "strategy": "bm25"
                })

        scores.sort(key=lambda x: -x["score"])
        return scores[:top_k]


class HybridRetriever:
    """Hybrid Retriever combining Dense & BM25 using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        # Candidate pool
        candidate_k = max(top_k * 4, 20)
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_k)
        bm25_results = self.bm25.search(query, filters=filters, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, item in enumerate(bm25_results):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: -rrf_scores[cid])
        fused_results = []
        for cid in sorted_cids[:top_k]:
            base = chunk_map[cid]
            fused_results.append({
                **base,
                "score": float(rrf_scores[cid]),
                "strategy": "hybrid_rrf"
            })
        return fused_results


class WeightedHybridRetriever:
    """Option 1: Adaptive Weighted Hybrid RRF with Entity/Lexical Prioritization.
    Boosts BM25 weight when queries contain proper nouns, capitalized entities, or quotes.
    """
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 30):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 4, 20)
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_k)
        bm25_results = self.bm25.search(query, filters=filters, top_k=candidate_k)

        # Detect entity / proper noun cues (capitalized words in query, quotes)
        has_entities = bool(re.search(r'[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]', query))
        w_bm25 = 0.88 if has_entities else 0.65
        w_dense = 0.12 if has_entities else 0.35

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_dense / (self.rrf_k + rank + 1)

        for rank, item in enumerate(bm25_results):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_bm25 / (self.rrf_k + rank + 1)

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: -rrf_scores[cid])
        fused = []
        for cid in sorted_cids[:top_k]:
            fused.append({
                **chunk_map[cid],
                "score": float(rrf_scores[cid]),
                "strategy": "weighted_hybrid"
            })
        return fused


class TwoStageRetriever:
    """Option 2: Two-Stage Retrieval (Candidate Retrieval N=15 -> Cross-Phrase Reranker down to Top-K).
    Fetches a broader pool (Top 15) and uses phrase matching, n-gram overlap & coverage to rerank.
    """
    def __init__(self, base_retriever: Any, candidate_k: int = 15):
        self.base = base_retriever
        self.candidate_k = candidate_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidates = self.base.search(query, filters=filters, top_k=max(self.candidate_k, top_k * 3))
        if not candidates:
            return []

        q_terms = normalize_text(query)
        q_tokens = set(q_terms)
        bigrams = [" ".join(q_terms[i:i+2]) for i in range(len(q_terms)-1)]
        trigrams = [" ".join(q_terms[i:i+3]) for i in range(len(q_terms)-2)]

        reranked = []
        for c in candidates:
            text_norm = " ".join(normalize_text(c["text"]))
            base_score = float(c.get("score", 0.0))

            phrase_bonus = 0.0
            for tri in trigrams:
                if tri in text_norm:
                    phrase_bonus += 2.5
            for bi in bigrams:
                if bi in text_norm:
                    phrase_bonus += 1.2

            c_tokens = set(normalize_text(c["text"]))
            coverage = len(q_tokens.intersection(c_tokens)) / max(len(q_tokens), 1)

            final_score = coverage * 2.0 + phrase_bonus * 1.5 + base_score
            reranked.append({
                **c,
                "score": final_score,
                "strategy": "two_stage_rerank"
            })

        reranked.sort(key=lambda x: -x["score"])
        return reranked[:top_k]
