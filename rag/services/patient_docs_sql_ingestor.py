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


#OBSOLETE - If you look at rag/views.py, the ingestion webhooks bypass this completely and just call qd_client.upsert() directly. This function is dead weight.
# =========================================
# 2. Store embedded document to vector database
# =========================================
# def ingest_patient_food_intake_doc(doc, embedded_doc):
#     # Generate a unique ID for this point
#     point_id = str(uuid.uuid4())  # UUID ensures no collision
    
#     # Upload document text + embedded document
#     point = PointStruct( 
#         id=point_id, # unique ID
#         vector=embedded_doc.tolist(),  # embedding as list
#         payload={
#             "text": doc, # store equivalent text
#             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#         }  
#     )

#     qd_client.upsert(collection_name=COLLECTION_NAME, points=[point])
