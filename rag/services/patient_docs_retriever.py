# THE LISTENER
# listener/services/5090_listener.py
# 1. Update 5070 MySQL table when 5090 MySQL table is updated
# test locally with food_intake_backend copy in 5070
# -> signals from 5090 food_intake_backend
# -> pass data to 5070 listener app endpoints
# -> update/delete 5070 MySQL tables accordingly

# AS RETRIEVER
# I. Document Ingestion 
# As a ltc_patient records a completion of a food intake, continuously ingest documents (e.g. patient intake records, patient DRI record) into vector database (i.e. Qdrant Collection “food_intakes”).

# 1. If a ltc_patient completes a food intake, create a document from retrieved from MySQL database (ltc_patient, food intake, meal, ingredients, nutrients)
# - services/patient_docs_creator.py
# -> Query 5070 MySQL table for relevant data
# -> Create a document in the format of:
# Patient Table information, Food Intake Table information, Meal Table information, Ingredients Table information, Nutrients Table information
# Must be in a relational format so that it can be easily parsed by LLM to construct patient context.

# 2. 
#  II. Retriever
# Given a prompt/query, retrieve relevant documents from vector database to build context for LLM.
# - services/patient_docs_retriever.py
# 1. Create prompt/query (based on exact datetime; categorized into (1) "recent"/Today, (2) Past week/past 7 days/past 6-2 day, (3) Past month
# - for Patient DRIs
# - for Patient Food Intake Records
# 2. Embed prompt/query
# 3. Using embedded prompt/query, retrieve relevant documents from vector database
# 4. Use retrieved documents to build context for LLM


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

qd_client = QdrantClient(host="localhost", port=6333)
VECTOR_SIZE = 384 # 384 is default embedding size
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


# =======================================
# Vector Search by Cosine Similarity
# =======================================
# English
def vector_search_patient_docs_english(query, top_k=5):
    # Qdrant Collection
    COLLECTION_NAME = "ltc_semantic_graph_2" 

    # Embed the query
    q_emb = EMBEDDING_MODEL.encode(query).tolist()

    # Search Qdrant (new API)
    results = qd_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k
    )

    return [(hit.payload["page_content"], hit.payload["metadata"], hit.score) for hit in results.points]


# Chinese
def vector_search_patient_docs_chinese(query, top_k=5):
    # Qdrant Collection
    COLLECTION_NAME = "ltc_chinese_semantic_graph" 

    # Embed the query
    q_emb = EMBEDDING_MODEL.encode(query).tolist()

    # Search Qdrant (new API)
    results = qd_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k
    )

    return [(hit.payload["page_content"], hit.payload["metadata"], hit.score) for hit in results.points]


# Query & Filter by patient ID (pid)
def vector_search_patient_docs(query, pid, top_k=5):
    # Qdrant Collection
    COLLECTION_NAME = "ltc_semantic_graph_2" 

    # Embed the query
    q_emb = EMBEDDING_MODEL.encode(query).tolist()

    # Search Qdrant (new API)
    results = qd_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k,
        query_filter = flt(pid)
    )

    return [(hit.payload["page_content"], hit.payload["metadata"], hit.score) for hit in results.points]


def get_patient_profile(pid):
    COLLECTION_NAME = "ltc_semantic_graph_2"

    results, _ = qd_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="metadata.doc_type", match=MatchValue(value="patient_profile")),
                FieldCondition(key="metadata.ltc_patient_id", match=MatchValue(value=pid)),
            ]
        ),
        limit=3
    )

    return [
        (point.payload["page_content"], point.payload.get("metadata"))
        for point in results
    ]

def get_patient_profile_by_room_and_bed(room_number, bed_number):
    COLLECTION_NAME = "ltc_semantic_graph_2"

    results, _ = qd_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="metadata.doc_type", match=MatchValue(value="patient_profile")),
                FieldCondition(key="metadata.room_number", match=MatchValue(value=room_number)),
                FieldCondition(key="metadata.bed_number", match=MatchValue(value=bed_number)),
            ]
        ),
        limit=3
    )

    return [
        (point.payload["page_content"], point.payload.get("metadata"))
        for point in results
    ]

def get_patient_dietary_targets(pid):
    COLLECTION_NAME = "ltc_semantic_graph_2"

    results, _ = qd_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=dri_flt(pid),
        limit=3
    )

    return [
        (point.payload["page_content"], point.payload.get("metadata"))
        for point in results
    ]


def get_patient_food_intake(pid, dts, limit_per_scroll=10):
    COLLECTION_NAME = "ltc_semantic_graph_2"

    all_results = []
    offset = None

    while True:
        scroll_result, next_offset = qd_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=bd_flt(pid, dts),
            limit=limit_per_scroll,
            offset=offset,
        )

        all_results.extend(scroll_result)

        # stop condition
        if next_offset is None:
            break

        offset = next_offset

    return [
        (point.payload.get("page_content"), point.payload.get("metadata"))
        for point in all_results
    ]


def get_patient_segmented_intake(pid, dts, limit_per_scroll=10):
    COLLECTION_NAME = "ltc_semantic_graph_2"

    all_results = []
    offset = None

    while True:
        scroll_result, next_offset = qd_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=si_flt(pid, dts),
            limit=limit_per_scroll,
            offset=offset,
        )

        all_results.extend(scroll_result)

        # stop condition
        if next_offset is None:
            break

        offset = next_offset

    return [
        (point.payload.get("page_content"), point.payload.get("metadata"))
        for point in all_results
    ]

# === Filters ===
def flt(pid):
    flt = Filter(
        must=[
            FieldCondition(key="metadata.doc_type", match=MatchValue(value="intake_event")),
            FieldCondition(key="metadata.ltc_patient_id", match=MatchValue(value=pid)),
        ]
    )
    return flt


def dri_flt(pid):
    return Filter(
        must=[
            FieldCondition(
                key="metadata.doc_type",
                match=MatchValue(value="dietary_target")
            ),
            FieldCondition(
                key="metadata.ltc_patient_id",
                match=MatchValue(value=pid)
            )
        ]
    )


def bd_flt(pid, date_str: str):
    return Filter(
        must=[
            FieldCondition(
                key="metadata.doc_type",
                match=MatchValue(value="intake_event")
            ),
            FieldCondition(
                key="metadata.ltc_patient_id",
                match=MatchValue(value=pid)
            ),
            FieldCondition(
                key="metadata.date",
                match=MatchValue(value=date_str)
            ),
        ]
    )


def si_flt(pid, date_str: str):
    return Filter(
        must=[
            FieldCondition(
                key="metadata.doc_type",
                match=MatchValue(value="segmented_intake")
            ),
            FieldCondition(
                key="metadata.ltc_patient_id",
                match=MatchValue(value=pid)
            ),
            FieldCondition(
                key="metadata.date",
                match=MatchValue(value=date_str)
            ),
        ]
    )


# ==================================
# Build Prompt
# ==================================

# FORMAT DOCUMENTS INTO CONTEXT
def format_food_intakes_docs(food_intake_res):
    return [
        {
            "result": i + 1,
            "document": doc,
            "metadata": metadata
        }
        for i, (doc, metadata) in enumerate(food_intake_res)
    ]

# BUILD PROMPT
def build_prompt_for_patient_docs(query, context):
    # Extract filenames from the context (assumes filenames are included in the context as part of the formatted data)
    sources = []
    
    # Create the formatted prompt
    return f"""
你是一個營養助理，請**只能根據下列資料**回答問題。

相關資料:
{context}

使用者問題:
{query}

回答規則:
- 只能使用「相關資料」中的內容
- 不可自行推論或補充
- 若資料中沒有明確答案，請回答「無資料」
"""


