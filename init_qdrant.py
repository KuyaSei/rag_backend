from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connect to your local Qdrant instance
client = QdrantClient(host="localhost", port=6333)

VECTOR_SIZE = 384  # Dimension for paraphrase-multilingual-MiniLM-L12-v2

print("Creating ltc_semantic_graph...")
client.recreate_collection(
    collection_name="ltc_semantic_graph",
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

print("Creating ltc_chinese_semantic_graph...")
client.recreate_collection(
    collection_name="ltc_chinese_semantic_graph",
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

print("Collections successfully initialized.")