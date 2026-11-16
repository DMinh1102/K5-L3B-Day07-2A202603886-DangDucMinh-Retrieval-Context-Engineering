"""Vector indexer for Day 7 Agentic RAG using fastembed (paraphrase-multilingual-MiniLM-L12-v2).
Runs 100% offline, zero rate limits, ultra-fast vector generation.
"""
import json
import time
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm


def build_embeddings(
    chunks_path: str | Path = "data/chunks.json",
    output_npy: str | Path = "data/embeddings.npy",
    meta_path: str | Path = "data/index_meta.json",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    chunks_path = Path(chunks_path)
    output_npy = Path(output_npy)
    meta_path = Path(meta_path)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Total chunks to embed: {len(chunks)} using {model_name}...")

    t0 = time.time()
    embedding_model = TextEmbedding(model_name=model_name)
    texts = [c["text"] for c in chunks]

    # Batch embed texts
    embeddings_list = []
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks locally"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = list(embedding_model.embed(batch_texts))
        embeddings_list.extend(batch_embeddings)

    vec_matrix = np.array(embeddings_list, dtype=np.float32)
    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vecs = vec_matrix / norms

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, normalized_vecs)
    total_sec = time.time() - t0
    print(f"Saved normalized matrix shape {normalized_vecs.shape} to {output_npy} in {total_sec:.2f}s")

    meta = {
        "model": model_name,
        "dimension": int(normalized_vecs.shape[1]),
        "total_chunks": len(chunks),
        "chunk_ids": [c["id"] for c in chunks],
        "source": "data/20260319_VNM_Bao_cao_thuong_nien_2025_eae619d5f6.pdf",
        "doc_id": "vnm-ar-2025",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved index metadata to {meta_path}")


if __name__ == "__main__":
    build_embeddings()
