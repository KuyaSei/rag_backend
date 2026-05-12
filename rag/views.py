from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import uuid

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# HPA RAG Services
from .services.hpa_retriever_services import build_prompt, build_rag_context, qdrant_search, retrieve_all

# Patient Docs RAG Services
from .services.patient_docs_sql_ingestor import embed_doc
from .services.patient_docs_retriever import get_patient_food_intake, get_patient_profile, get_patient_segmented_intake, vector_search_patient_docs_chinese, vector_search_patient_docs_english, build_prompt_for_patient_docs, vector_search_patient_docs
from .services.generator import ask_llm

from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from .services.patient_docs_sql_ingestor import embed_doc, qd_client




class CombinedRAGView(APIView):
    def post(self, request):
        query = request.data.get("query")

        if not query:
            return Response( {"detail": "Missing 'query' in request data."}, status=status.HTTP_400_BAD_REQUEST )

        try:
            # Retrieve chunks from 2 knowledge base
            patient_docs_results = vector_search_patient_docs_english(query, top_k=5)
            hpa_docs_results = retrieve_all(query, top_k=10)

            # Chunks
            patient_docs_chunks = [
                {
                    "result": i + 1,
                    "score": score,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata, score) in enumerate(patient_docs_results)
            ]


            # Build context 
            patient_docs_context = "\n\n".join([doc for doc, metadata, score in patient_docs_results])
            hpa_docs_context = build_rag_context(hpa_docs_results)

            # Build prompt
            prompt=f"""
Use the provided context only when appropriate.

相關資訊：
Patient context":
{patient_docs_context}

Dietary guidelines context:
{hpa_docs_context}

用戶提問：
{query}

回應規則：
- *Important* If the 用戶提問 or query is not about a specific patient do not use the Patient context.
- *Important* If the 用戶提問 or query is not dietary guidelines related do not use the Dietary guidelines context.
- 僅使用「相關資訊」區塊的內容
- 請僅以繁體中文回覆。
"""

            # 4. Pass context
            answer = ask_llm(prompt)

            return Response( {
                "response": answer,
                "patient_docs_chunks": patient_docs_chunks,
                "hpa_docs_chunks": hpa_docs_results
                }, 
                status=status.HTTP_200_OK )
        
        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response from RAG.",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Taiwan HPA RAG
class HpaDocsRetrievalRagQueryView(APIView):
    def post(self, request):
        query = request.data.get("query")

        if not query:
            return Response( {"detail": "Missing 'query' in request data."}, status=status.HTTP_400_BAD_REQUEST )

        try:
            # 1. Retrieve chunks
            chunks = retrieve_all(query, top_k=5)

            # 2. Build context
            context = build_rag_context(chunks)

            # 3. Build prompt
            prompt = build_prompt(query, context)

            # 4. Pass context
            answer = ask_llm(prompt)
            return Response( {
                "response": answer,
                "retrieved_chunks_and_scores": chunks 
                }, 
                status=status.HTTP_200_OK )
        
        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response from RAG.",
                    "error": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# (CHINESE) Patient Docs RAG        
class PatientDocsRagQueryView(APIView):
    def post(self, request):
        query = request.data.get("query")

        if not query:
            return Response( {"detail": "Missing 'query' in request data."}, status=status.HTTP_400_BAD_REQUEST )

        try:
            # 1. Retrieve chunks
            results = vector_search_patient_docs_chinese(query, top_k=5)

            # 2. Chunks
            chunks = [
                {
                    "result": i + 1,
                    "score": score,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata, score) in enumerate(results)
            ]


            # 2. Build context 
            context = "\n\n".join([doc for doc, metadata, score in results])

            # 3. Build prompt
            prompt=f"""
請僅基於以下提供的資訊回答問題。

相關資訊：
{context}

用戶提問：
{query}

回應規則：
- 僅使用「相關資訊」區塊的內容
- 不得進行推論或添加額外資訊
- 若資訊中無明確答案，請回應「無可用資訊」
"""

            # 4. Pass context
            answer = ask_llm(prompt)

            return Response( {
                "gen_response": answer,
                "retrieved_chunks": chunks,
                "context": context
                }, 
                status=status.HTTP_200_OK )
        
        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response from RAG.",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# (ENGLISH) Patient Docs RAG
class PatientDocsEnglishRagQueryView(APIView):
    def post(self, request):
        query = request.data.get("query")

        if not query:
            return Response( {"detail": "Missing 'query' in request data."}, status=status.HTTP_400_BAD_REQUEST )

        try:
            # 1. Retrieve chunks
            results = vector_search_patient_docs_english(query, top_k=5)

            # 2. Chunks
            chunks = [
                {
                    "result": i + 1,
                    "score": score,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata, score) in enumerate(results)
            ]

            # 2. Build context 
            context = "\n\n".join([doc for doc, metadata, score in results])

            # 3. Build prompt
            prompt=f"""
請僅基於以下提供的資訊回答問題。

相關資訊：
{context}

用戶提問：
{query}

回應規則：
- 僅使用「相關資訊」區塊的內容
- 不得進行推論或添加額外資訊
- 若資訊中無明確答案，請回應「無可用資訊」
- 請僅以繁體中文回覆。
"""

            # 4. Pass context
            answer = ask_llm(prompt)

            return Response( {
                "gen_response": answer,
                "retrieved_chunks": chunks,
                "context": context
                }, 
                status=status.HTTP_200_OK )
        
        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response from RAG.",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



# 5090 server
class Receive5090PayloadView(APIView):
    """
    Webhook to catch semantic JSON payloads from the 5090, 
    embed them using the model, and store them in an isolated Qdrant collection.
    """
    def post(self, request):
        try:
            data = request.data
            page_content = data.get("page_content")
            metadata = data.get("metadata", {})

            if not page_content:
                return Response({"detail": "Missing page_content"}, status=status.HTTP_400_BAD_REQUEST)

            # 1. Embed the text using the existing SentenceTransformer model
            embedded_doc = embed_doc(page_content)

            # 2. Create a deterministic ID (so updates overwrite the old vector)

            # doc_type = metadata.get("doc_type", "unknown")
            # room = metadata.get("room_number", "0")
            # bed = metadata.get("bed_number", "0")
            # date_str = metadata.get("date", "")
            # phase = metadata.get("meal_phase", "")
            
            # unique_string = f"{doc_type}_{room}_{bed}_{date_str}_{phase}"
            # # Generate consistent UUID based on the unique string
            # point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))


            # NEW ID generation logic based on doc_type
            # 1000 => intake_event
            # 2000 => patient_profile
            # 3000 => dietary_target
            # 4000 => segmented_intake
            doc_type = metadata.get("doc_type", "unknown")

            if doc_type == "patient_profile":
                point_id = int(f"2000{metadata.get('ltc_patient_id')}")

            elif doc_type == "dietary_target":
                point_id = int(f"3000{metadata.get('ltc_patient_id')}")

            elif doc_type == "intake_event":
                point_id = int(f"1000{metadata.get('intake_id')}")

            elif doc_type == "segmented_intake":
                point_id = int(f"4000{metadata.get('estimation_id')}")

            else:
                import uuid
                point_id = uuid.uuid4().int >> 64  # fallback integer

            # 3. Save to a NEW isolated Qdrant collection
            point = PointStruct(
                id=point_id,
                vector=embedded_doc.tolist(),
                payload={
                    "page_content": page_content,
                    "metadata": metadata
                }
            )

            qd_client.upsert(
                collection_name="ltc_semantic_graph_2", 
                points=[point]
            )

            print(f"Upserted point with ID: {point_id} into ltc_semantic_graph 2 collection.")

            return Response({"status": "Successfully ingested to 5070 Qdrant", "id": point_id}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class Receive5090PayloadChineseDocsView(APIView):
    """ CHINESE VERSION
    Webhook to catch semantic JSON payloads from the 5090, 
    embed them using the model, and store them in an isolated Qdrant collection.
    """
    def post(self, request):
        try:
            data = request.data
            page_content = data.get("page_content")
            metadata = data.get("metadata", {})

            if not page_content:
                return Response({"detail": "Missing page_content"}, status=status.HTTP_400_BAD_REQUEST)

            # 1. Embed the text using the existing SentenceTransformer model
            embedded_doc = embed_doc(page_content)

            # 2. Create a deterministic ID (so updates overwrite the old vector)

            # doc_type = metadata.get("doc_type", "unknown")
            # room = metadata.get("room_number", "0")
            # bed = metadata.get("bed_number", "0")
            # date_str = metadata.get("date", "")
            # phase = metadata.get("meal_phase", "")
            # unique_string = f"{doc_type}_{room}_{bed}_{date_str}_{phase}"
            # # Generate consistent UUID based on the unique string
            # point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))


            # NEW ID generation logic based on doc_type
            # 1000 => intake_event
            # 2000 => patient_profile
            # 3000 => dietary_target
            # 4000 => segmented_intake
            doc_type = metadata.get("doc_type", "unknown")

            if doc_type == "patient_profile":
                point_id = int(f"2000{metadata.get('ltc_patient_id')}")

            elif doc_type == "dietary_target":
                point_id = int(f"3000{metadata.get('ltc_patient_id')}")

            elif doc_type == "intake_event":
                point_id = int(f"1000{metadata.get('intake_id')}")

            elif doc_type == "segmented_intake":
                point_id = int(f"4000{metadata.get('estimation_id')}")

            else:
                import uuid
                point_id = uuid.uuid4().int >> 64  # fallback integer

            # 3. Save to a NEW isolated Qdrant collection
            point = PointStruct(
                id=point_id,
                vector=embedded_doc.tolist(),
                payload={
                    "page_content": page_content,
                    "metadata": metadata
                }
            )

            qd_client.upsert(
                collection_name="ltc_chinese_semantic_graph", 
                points=[point]
            )

            return Response({"status": "Successfully ingested to 5070 Qdrant", "id": point_id}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PatientFoodIntakeSummaryView(APIView):    
    def get(self, request, pk):
        try:
            patient_profile= get_patient_profile(pk)
            patient = patient_profile[0][1]
            # curdate = datetime.now().strftime("%Y-%m-%d")
            curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

            # Get relevant docs
            food_intake_res = get_patient_food_intake(pk, curdate,)

            if not food_intake_res:
                return Response({
                    "response": "No food intake records were recorded for this date.",
                    "final_prompt": None,
                    "food_intake_chunks": "",
                    "date": curdate
                }, status=status.HTTP_200_OK)

            food_intake_chunks = [
                {
                    "result": i + 1,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata) in enumerate(food_intake_res)
            ]

            # Build context
            food_intake_context = "\n\n ".join([doc for doc, metadata in food_intake_res])

            # Build prompt
            prompt = (
                f"Based on the following food intake records for {curdate}, write a brief summary for a long-term care patient."
                f"\n\nFood intake records:\n{food_intake_context}"
                f"\n\nRules:"
                f"\n1. State ONLY which meals the patient had (e.g., lunch, dinner, both, or none)."
                f"\n2. For each meal present, mention only whether a before-meal (Gross) and after-meal (Leftover/Net) record was taken. Do NOT mention any specific quantities, grams, or milliliters."
                f"\n3. If no records exist for {curdate}, state: 'No food intake records were recorded for this date.'"
                f"\n4. Keep the response to 2-3 sentences maximum. Answer in English."
            )

            # Pass context to LLM
            response = ask_llm(prompt)

            return Response(
                {
                    "response": response,
                    "final_prompt": prompt,
                    "food_intake_chunks": food_intake_context,
                    "date": curdate
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response", 
                    "error": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PatientFoodIntakeSummaryByDateView(APIView):    
    def get(self, request, pk, date):
        try:
            patient_profile= get_patient_profile(pk)
            patient = patient_profile[0][1]

            # Get relevant docs
            food_intake_res = get_patient_food_intake(pk, date)

            if not food_intake_res:
                return Response({
                    "response": "No food intake records were recorded for this date.",
                    "final_prompt": None,
                    "food_intake_chunks": "",
                    "food_intake_res": [],
                    "date": date
                }, status=status.HTTP_200_OK)

            food_intake_chunks = [
                {
                    "result": i + 1,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata) in enumerate(food_intake_res)
            ]

            # Build context
            food_intake_context = "\n\n ".join([doc for doc, metadata in food_intake_res])

            # Build prompt
            prompt = (
                f"Based on the following food intake records for {date}, write a brief summary for a long-term care patient."
                f"\n\nFood intake records:\n{food_intake_context}"
                f"\n\nRules:"
                f"\n1. State ONLY which meals the patient had (e.g., lunch, dinner, both, or none)."
                f"\n2. For each meal present, mention only whether a before-meal (Gross) and after-meal (Leftover/Net) record was taken. Do NOT mention any specific quantities, grams, or milliliters."
                f"\n3. If no records exist for {date}, state: 'No food intake records were recorded for this date.'"
                f"\n4. Keep the response to 2-3 sentences maximum. Answer in English."
            )

            # Pass context to LLM
            response = ask_llm(prompt)

            return Response(
                {
                    "response": response,
                    "final_prompt": prompt,
                    "food_intake_chunks": food_intake_context,
                    "food_intake_res": food_intake_res,
                    "date": date
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response", 
                    "error": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




