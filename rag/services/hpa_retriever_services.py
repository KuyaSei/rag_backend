import json
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Chinese tokenizer (for TF-IDF)
from .tokenizers import jieba_tokenizer

#==============================================
# 1. Load Vectorizers
#==============================================

# 1.1. Load embedding model (SEMANTIC SEARCH)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME, device="cpu")


# 1.2. Load Vectorizer (LEXICAL SEARCH)

import os
import pickle
import scipy.sparse as sp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(CURRENT_DIR, "tfidf_vectorizers")

def load_pickle(filename):
    path = os.path.join(VECTOR_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

tfidf_vectorizer_table = load_pickle("tfidf_vectorizer_table.pkl")
tfidf_vectorizer_mixed = load_pickle("tfidf_vectorizer_mixed.pkl")
tfidf_vectorizer_text  = load_pickle("tfidf_vectorizer_text.pkl")



#==============================================
# 2. Load Vector Databases
#==============================================

# 2.1. Dense Vector Collections

# Initialize Qdrant client
qd_client = QdrantClient(url="http://localhost:6333")
# Dense Vector Collections
hpa_collections = ["qdrant_table_chunks", "qdrant_mixed_chunks", "qdrant_text_chunks"] 


# 2.2.Sparse TF-IDF matrices

MATRIX_DIR = os.path.join(CURRENT_DIR, "tfidf_matrices")

tfidf_matrix_table = sp.load_npz( os.path.join(MATRIX_DIR, "tfidf_matrix_table.npz"))
tfidf_matrix_mixed = sp.load_npz(os.path.join(MATRIX_DIR, "tfidf_matrix_mixed.npz"))
tfidf_matrix_text = sp.load_npz(os.path.join(MATRIX_DIR, "tfidf_matrix_text.npz"))


# Metadata for TF-IDF entries
META_DIR = os.path.join(CURRENT_DIR, "meta_text_chunks")
with open(os.path.join(META_DIR, "table_heavy_chunks.json"), "r", encoding="utf-8") as f:
    table_meta = json.load(f)

with open(os.path.join(META_DIR, "mixed_chunks.json"), "r", encoding="utf-8") as f:
    mixed_meta = json.load(f)

with open(os.path.join(META_DIR, "text_only_chunks.json"), "r", encoding="utf-8") as f:
    text_meta = json.load(f)



#==============================================
# 3. Retriever Functions
#==============================================

# 3.1. Semantic Search; Cosine Similarity Search in Qdrant Collection

def qdrant_search(collection_name, query_text, top_k=5, query_filter=None):
    # Encode/embed query 
    query_vector = model.encode( query_text, normalize_embeddings=True ).astype("float32").tolist()

    response = qd_client.query_points(
        collection_name = collection_name,
        query = query_vector,
        limit = top_k,
        query_filter = query_filter,
        with_payload=True
    )

    results = []
    for point in response.points:
        results.append({
            "semantic_score": point.score,
            "text": point.payload.get("text", ""),
            "metadata": {k: v for k, v in point.payload.items() if k != "text"}
        })

    return results


# 3.2. Lexical Search (TF-IDF); Cosine Similarity Search in TF-IDF Matrix

from sklearn.metrics.pairwise import cosine_similarity

def tfidf_search(vectorizer, tfidf_matrix, meta, query, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "lexical_score": float(scores[idx]),
            "text": meta[idx]["text"],
            "metadata": meta[idx]["metadata"]
        })
    return results


#==============================================
# 4. Scoring and Hybrid Retrieval
#==============================================
# 4.1. Score Normalization
def normalize_scores(results, key):
    scores = [r[key] for r in results if key in r]
    if not scores:
        return results

    min_s = min(scores)
    max_s = max(scores)

    if max_s - min_s == 0:
        for r in results:
            r[key] = 0.0
        return results

    for r in results:
        if key in r:
            r[key] = (r[key] - min_s) / (max_s - min_s)

    return results


# 4.2. Hybrid Retrieval

# QDRANT COLLECTION NAMES FOR HPA DOCUMENTS
qdrant_collection_table_ = "qdrant_table_chunks"
qdrant_collection_mixed = "qdrant_mixed_chunks"
qdrant_collection_text  = "qdrant_text_chunks"

def retrieve_all(query, top_k=5):
    results = []

    sources = [
        ("table", qdrant_collection_table_, table_meta, tfidf_vectorizer_table, tfidf_matrix_table),
        ("mixed", qdrant_collection_mixed, mixed_meta, tfidf_vectorizer_mixed, tfidf_matrix_mixed),
        ("text",  qdrant_collection_text,  text_meta,  tfidf_vectorizer_text,  tfidf_matrix_text),
    ]

    for source_name, collection_name, meta, vectorizer, tfidf_matrix in sources:

        semantic_hits = qdrant_search(collection_name, query, top_k=20)
        lexical_hits  = tfidf_search(vectorizer, tfidf_matrix, meta, query, top_k=20)

        lexical_dict = {r["text"]: r["lexical_score"] for r in lexical_hits}

        for r in semantic_hits:
            r["source"] = source_name
            r["lexical_score"] = lexical_dict.get(r["text"], 0.0)
            results.append(r)

    results = normalize_scores(results, "semantic_score")
    results = normalize_scores(results, "lexical_score")

    for r in results:
        r["score"] = 0.6 * r["semantic_score"] + 0.4 * r["lexical_score"]

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# STANDBY - used only by the monthly recommender which is bypassed by frontend
def retrieve_text(query, top_k=5):
    results = []

    sources = [
        ("text",  qdrant_collection_text,  text_meta,  tfidf_vectorizer_text,  tfidf_matrix_text),
    ]

    for source_name, collection_name, meta, vectorizer, tfidf_matrix in sources:

        semantic_hits = qdrant_search(collection_name, query, top_k=20)
        lexical_hits  = tfidf_search(vectorizer, tfidf_matrix, meta, query, top_k=20)

        lexical_dict = {r["text"]: r["lexical_score"] for r in lexical_hits}

        for r in semantic_hits:
            r["source"] = source_name
            r["lexical_score"] = lexical_dict.get(r["text"], 0.0)
            results.append(r)

    # Normalize globally
    results = normalize_scores(results, "semantic_score")
    results = normalize_scores(results, "lexical_score")

    # Combine scores
    for r in results:
        r["score"] = 0.6 * r["semantic_score"] + 0.4 * r["lexical_score"]

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# 4.3. RAG Context Building
def build_rag_context(chunks):
    context_blocks = []

    for i, c in enumerate(chunks, 1):
        header = f"""【Result {i}】(Source: {c['metadata'].get('filename', 'Unknown')})\n"""

        body = c["text"].strip()

        context_blocks.append(header + body)

    return "\n\n".join(context_blocks)

# STANDBY - A more complex context builder that combines both the HPA and patient intake retrieval results into one prompt, with clear section headers. Used by the CombinedRAGView endpoint in rag/views.py, which is a general-purpose RAG endpoint that isn't currently tied to any specific UI component but could be used for a general "Ask the AI" feature.

#==============================================
# 5. Generation
#==============================================

# 5.1. Prompt
def build_prompt(query, context):
    # Extract filenames from the context (assumes filenames are included in the context as part of the formatted data)
    sources = []
    
    # Parse the context and extract filenames
    lines = context.split("\n")
    for line in lines:
        if "檔案:" in line:
            # Extract filename (everything after "檔案:")
            filename = line.split("檔案:")[1].strip()
            sources.append(filename)
    
    # Create the formatted prompt
    return f"""
你是一個營養助理，請**只能根據下列資料**回答問題。

相關資料:
{context}

使用者問題:
{query}

回答規則:
- 不可自行推論或補充
- 若資料中沒有明確答案，請回答「無資料」

來源:
{', '.join(sources)}  
"""

# Build Combined Prompt
def build_combined_rag_context(hpa_chunks, intake_chunks):
    sections = []

    if intake_chunks:
        intake_blocks = []
        for i, c in enumerate(intake_chunks, 1):
            intake_blocks.append(header + "\n" + c["text"].strip())
        sections.append("=======\n" + "\n\n".join(intake_blocks))

    if hpa_chunks:
        hpa_blocks = []
        for i, c in enumerate(hpa_chunks, 1):
            meta = c["metadata"]
            header = f"【HPA資料 {i}】來源: {c['source']} | 檔案: {meta.get('filename','未知')} | 頁: {meta.get('page','未知')} | 分數: {c['score']:.4f}"
            hpa_blocks.append(header + "\n" + c["text"].strip())
        sections.append("======\n" + "\n\n".join(hpa_blocks))

    return "\n\n".join(sections)


# 5.2. LLM
# rag/services/generator.py
