"""Automated benchmark runner for Day 7 Agentic RAG.
Runs all retrieval experiments on Dev, compares improvements, evaluates Agent,
and evaluates the selected pipeline on the Test set.
"""
import json
import os
import sys
import time
from pathlib import Path

# Add starter to path
sys.path.insert(0, str(Path(__file__).parent / "starter"))

from dotenv import load_dotenv
from google import genai

from baseline import hit, pack_context
from agentic import run_agent
from agent_metrics import aggregate
from src.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from src.agent import GeminiController
from src.evaluate import evaluate_retriever, evaluate_agent


def run_all():
    load_dotenv()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
    dev_q = [q for q in questions if q["split"] == "dev"]
    test_q = [q for q in questions if q["split"] == "test"]
    cases = json.loads(Path("data/cases.json").read_text(encoding="utf-8"))

    print(f"Loaded {len(dev_q)} Dev questions and {len(test_q)} Test questions.")

    # 1. Initialize retrievers
    print("Initializing retrievers...")
    bm25 = BM25Retriever("data/chunks.json")
    dense = DenseRetriever("data/chunks.json", "data/embeddings.npy")
    hybrid = HybridRetriever(dense, bm25, rrf_k=60)

    # 2. Retrieval Experiments on Dev
    retrieval_summaries = {}

    print("\n--- Running Dev: Dense Retrieval (Baseline) ---")
    t_dense, s_dense = evaluate_retriever(dense, dev_q, budget=1200)
    retrieval_summaries["dense_dev"] = s_dense
    (results_dir / "dense_dev_traces.json").write_text(json.dumps(t_dense, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Dense Dev: Hit@1: {s_dense['hit1']}/{s_dense['n']} ({s_dense['hit1_rate']:.1%}), Hit@3: {s_dense['hit3']}/{s_dense['n']} ({s_dense['hit3_rate']:.1%}), Context Evidence: {s_dense['context_evidence_rate']:.1%}")

    print("\n--- Running Dev: BM25 Retrieval ---")
    t_bm25, s_bm25 = evaluate_retriever(bm25, dev_q, budget=1200)
    retrieval_summaries["bm25_dev"] = s_bm25
    (results_dir / "bm25_dev_traces.json").write_text(json.dumps(t_bm25, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"BM25 Dev: Hit@1: {s_bm25['hit1']}/{s_bm25['n']} ({s_bm25['hit1_rate']:.1%}), Hit@3: {s_bm25['hit3']}/{s_bm25['n']} ({s_bm25['hit3_rate']:.1%}), Context Evidence: {s_bm25['context_evidence_rate']:.1%}")

    print("\n--- Running Dev: Hybrid RRF Retrieval ---")
    t_hybrid, s_hybrid = evaluate_retriever(hybrid, dev_q, budget=1200)
    retrieval_summaries["hybrid_dev"] = s_hybrid
    (results_dir / "hybrid_dev_traces.json").write_text(json.dumps(t_hybrid, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Hybrid Dev: Hit@1: {s_hybrid['hit1']}/{s_hybrid['n']} ({s_hybrid['hit1_rate']:.1%}), Hit@3: {s_hybrid['hit3']}/{s_hybrid['n']} ({s_hybrid['hit3_rate']:.1%}), Context Evidence: {s_hybrid['context_evidence_rate']:.1%}")

    # Improvement 1: Context Budget Expansion to 2000 chars
    print("\n--- Running Dev Improvement 1: Hybrid RRF + Budget 2000 chars ---")
    t_imp1, s_imp1 = evaluate_retriever(hybrid, dev_q, budget=2000)
    retrieval_summaries["imp1_dev"] = s_imp1
    (results_dir / "imp1_dev_traces.json").write_text(json.dumps(t_imp1, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Imp 1 Dev: Hit@1: {s_imp1['hit1']}/{s_imp1['n']} ({s_imp1['hit1_rate']:.1%}), Hit@3: {s_imp1['hit3']}/{s_imp1['n']} ({s_imp1['hit3_rate']:.1%}), Context Evidence: {s_imp1['context_evidence_rate']:.1%}")

    # Improvement 2: Hybrid RRF with RRF-k tuning (k=30)
    print("\n--- Running Dev Improvement 2: Hybrid RRF with rrf_k=30 + Budget 2000 ---")
    hybrid_k30 = HybridRetriever(dense, bm25, rrf_k=30)
    t_imp2, s_imp2 = evaluate_retriever(hybrid_k30, dev_q, budget=2000)
    retrieval_summaries["imp2_dev"] = s_imp2
    (results_dir / "imp2_dev_traces.json").write_text(json.dumps(t_imp2, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Imp 2 Dev: Hit@1: {s_imp2['hit1']}/{s_imp2['n']} ({s_imp2['hit1_rate']:.1%}), Hit@3: {s_imp2['hit3']}/{s_imp2['n']} ({s_imp2['hit3_rate']:.1%}), Context Evidence: {s_imp2['context_evidence_rate']:.1%}")

    # 3. Agent Experiments on Dev
    print("\n--- Running Agent Dev: Hybrid RRF + Budget 2000 ---")
    controller = GeminiController(model_name="gemini-3.5-flash-lite")
    agent_dev_traces, agent_dev_reviews, agent_dev_summary = evaluate_agent(
        controller=controller,
        retriever=hybrid,
        questions=dev_q,
        budget=2000,
        max_searches=3,
        top_k=3
    )
    (results_dir / "agent_dev_traces.json").write_text(json.dumps(agent_dev_traces, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "agent_dev_reviews.json").write_text(json.dumps(agent_dev_reviews, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "agent_dev_summary.json").write_text(json.dumps(agent_dev_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Agent Dev Summary:")
    print(json.dumps(agent_dev_summary, indent=2, ensure_ascii=False))

    # 4. Final Evaluation on TEST split (after fixing configuration)
    print("\n==========================================")
    print("RUNNING FINAL EVALUATION ON TEST SET (30 questions)")
    print("==========================================")

    # Test: Dense Baseline
    print("\n--- Running Test: Dense Retrieval Baseline ---")
    t_dense_test, s_dense_test = evaluate_retriever(dense, test_q, budget=1200)
    retrieval_summaries["dense_test"] = s_dense_test
    print(f"Dense Test: Hit@1: {s_dense_test['hit1']}/{s_dense_test['n']} ({s_dense_test['hit1_rate']:.1%}), Hit@3: {s_dense_test['hit3']}/{s_dense_test['n']} ({s_dense_test['hit3_rate']:.1%}), Context Evidence: {s_dense_test['context_evidence_rate']:.1%}")

    # Test: Chosen Pipeline (Hybrid RRF + Budget 2000)
    print("\n--- Running Test: Chosen Hybrid Retrieval ---")
    t_hybrid_test, s_hybrid_test = evaluate_retriever(hybrid, test_q, budget=2000)
    retrieval_summaries["hybrid_test"] = s_hybrid_test
    print(f"Hybrid Test: Hit@1: {s_hybrid_test['hit1']}/{s_hybrid_test['n']} ({s_hybrid_test['hit1_rate']:.1%}), Hit@3: {s_hybrid_test['hit3']}/{s_hybrid_test['n']} ({s_hybrid_test['hit3_rate']:.1%}), Context Evidence: {s_hybrid_test['context_evidence_rate']:.1%}")

    # Test: Full Agent on Test Set
    print("\n--- Running Full Agent on Test Set ---")
    agent_test_traces, agent_test_reviews, agent_test_summary = evaluate_agent(
        controller=controller,
        retriever=hybrid,
        questions=test_q,
        budget=2000,
        max_searches=3,
        top_k=3
    )
    (results_dir / "agent_test_traces.json").write_text(json.dumps(agent_test_traces, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "agent_test_reviews.json").write_text(json.dumps(agent_test_reviews, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "agent_test_summary.json").write_text(json.dumps(agent_test_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nAgent Test Summary:")
    print(json.dumps(agent_test_summary, indent=2, ensure_ascii=False))

    # Save overall retrieval summaries
    (results_dir / "retrieval_summaries.json").write_text(json.dumps(retrieval_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nAll experiments completed! Results saved to results/")


if __name__ == "__main__":
    run_all()
