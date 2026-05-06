from datetime import date, datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid

qd_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "food_intakes_vector_db"
VECTOR_SIZE = 384 # 384 is default embedding size
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# =========================================
# 1. Embed document
# =========================================
def embed_doc(doc):
    emb = EMBEDDING_MODEL.encode(doc)
    return emb


