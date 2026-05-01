import os, json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. Initialize DB and Embedding Model
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

# 2. Define the Collections to create
collections = [
    ("qdrant_table_chunks", "rag/services/meta_text_chunks/table_heavy_chunks.json"),
    ("qdrant_mixed_chunks", "rag/services/meta_text_chunks/mixed_chunks.json"),
    ("qdrant_text_chunks", "rag/services/meta_text_chunks/text_only_chunks.json")
]

print("Starting HPA Data Upload to Qdrant...")

for collection_name, json_path in collections:
    print(f"-> Recreating collection: {collection_name}")
    
    # Create the Collection in Qdrant
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Load JSON and format it into Qdrant Points
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    points = []
    print(f"Preparing {len(chunks)} chunks for {collection_name}...")
    for idx, item in enumerate(chunks):
        # Embed the text
        vector = model.encode(item["text"], normalize_embeddings=True).astype("float32").tolist()
        
        # Merge text with metadata so Qdrant can store it all together
        payload = {"text": item["text"]}
        payload.update(item["metadata"])
        
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    # Push to local Qdrant
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Success! Uploaded {len(points)} documents to {collection_name}.\n")

print("Phase 3A: Taiwan HPA Guidelines Successfully Uploaded!")