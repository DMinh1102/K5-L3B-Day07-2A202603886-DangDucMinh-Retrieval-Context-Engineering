"""Experiment and evaluation runner for Day 7 Agentic RAG.
Runs retrieval experiments (Dense, BM25, Hybrid RRF, Rerank) and Agent experiments.
Produces standard traces.json, reviews.json, and summary.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure starter is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter"))

from dotenv import load_dotenv
from google import genai

from baseline import hit, pack_context
from agentic import run_agent
from agent_metrics import aggregate
from src.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from src.agent import GeminiController


def evaluate_retriever(retriever: Any, questions: list[dict], budget: int = 1200) -> tuple[list[dict], dict]:
    traces = []
    started_all = time.perf_counter()

    for q in questions:
        if q.get("answerable") is False:
            continue
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=q.get("filter"), top_k=3)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000

        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        hit1 = bool(marked and marked[0]["hit"])
        hit3 = any(c["hit"] for c in marked)
        context_hit = any(c["hit"] and c["id"] in ids for c in marked)

        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit
        })

    total_time = time.perf_counter() - started_all
    n = len(traces)
    summary = {
        "n": n,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": sum(t["hit1"] for t in traces) / (n or 1),
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": sum(t["hit3"] for t in traces) / (n or 1),
        "context_evidence_rate": sum(t["context_hit"] for t in traces) / (n or 1),
        "avg_context_chars": sum(t["context_chars"] for t in traces) / (n or 1),
        "avg_latency_ms": sum(t["elapsed_ms"] for t in traces) / (n or 1),
        "total_seconds": total_time
    }
    return traces, summary


def evaluate_agent(
    controller: GeminiController,
    retriever: Any,
    questions: list[dict],
    budget: int = 1500,
    max_searches: int = 3,
    top_k: int = 3
) -> tuple[list[dict], list[dict], dict]:
    traces = []
    reviews = []
    chunk_map = {c["id"]: c for c in json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))}

    for q in questions:
        qid = q["id"]
        is_answerable = q.get("answerable", True)
        start = time.perf_counter()

        try:
            agent_out = run_agent(
                question=q["question"],
                controller=controller,
                retriever=retriever,
                filters=q.get("filter"),
                max_searches=max_searches,
                top_k=top_k,
                budget=budget
            )
            status = agent_out.get("status", "abstain")
            answer = agent_out.get("answer", "")
            citations = agent_out.get("citations", [])
            events = agent_out.get("events", [])
        except Exception as e:
            status = "error"
            answer = f"Error: {e}"
            citations = []
            events = [{"event": "error", "error": str(e)}]

        elapsed_ms = (time.perf_counter() - start) * 1000

        traces.append({
            "query_id": qid,
            "status": status,
            "answer": answer,
            "citations": citations,
            "events": events,
            "elapsed_ms": elapsed_ms
        })

        # Grounding Audit (Review)
        if status == "error":
            continue

        citation_support = []
        for cit_id in citations:
            # Check if citation actually exists and supports the statement
            c_chunk = chunk_map.get(cit_id)
            if not c_chunk:
                citation_support.append(False)
            elif is_answerable:
                # Does this chunk contain the gold evidence or keywords?
                has_gold = any(g["evidence"] in c_chunk["text"] for g in q.get("gold", []))
                citation_support.append(has_gold)
            else:
                citation_support.append(False)

        if is_answerable:
            if status == "abstain":
                grade = 0
            else:
                # Grade grounded answer 0, 1, 2
                gold_evidences = [g["evidence"] for g in q.get("gold", [])]
                found_evidences = sum(1 for ev in gold_evidences if any(ev in chunk_map.get(cid, {}).get("text", "") for cid in citations))
                if found_evidences == len(gold_evidences) and len(citations) > 0 and all(citation_support):
                    grade = 2
                elif found_evidences > 0:
                    grade = 1
                else:
                    grade = 0
            reviews.append({
                "query_id": qid,
                "answer_grade": grade,
                "citation_support": citation_support
            })
        else:
            abstention_correct = (status == "abstain")
            reviews.append({
                "query_id": qid,
                "abstention_correct": abstention_correct,
                "citation_support": citation_support
            })
        time.sleep(1.2)  # Avoid Gemini API rate limit

    cases = [
        {"id": q["id"], "split": q["split"], "answerable": q.get("answerable", True)}
        for q in questions
    ]
    summary = aggregate(cases, traces, reviews)
    return traces, reviews, summary
