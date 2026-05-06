from django.urls import path
from .views import CombinedRAGView, HpaDocsRetrievalRagQueryView, PatientDocsRagQueryView, PatientDocsEnglishRagQueryView, PatientFoodIntakeSummaryByDateView, PatientFoodIntakeSummaryView, Receive5090PayloadView, Receive5090PayloadChineseDocsView

# BASE ENDPOINT: /api/z

urlpatterns = [
    # Combined RAG
    path("rag/query/", CombinedRAGView.as_view()),
    path("rag/query", CombinedRAGView.as_view()),

    # Taiwan HPA RAG
    path("rag/hpa/query/", HpaDocsRetrievalRagQueryView.as_view()),
    path("rag/hpa/query", HpaDocsRetrievalRagQueryView.as_view()),

    # (CHINESE) Patient Docs RAG
    path("rag/patient/query/", PatientDocsRagQueryView.as_view()),
    path("rag/patient/query", PatientDocsRagQueryView.as_view()),

    # (ENGLISH) Patient Docs RAG
    path("rag/patient/en/query/", PatientDocsEnglishRagQueryView.as_view()),
    path("rag/patient/en/query", PatientDocsEnglishRagQueryView.as_view()),

    # Patient Daily food intake summary
    path("patient/<int:pk>/meal-intake", PatientFoodIntakeSummaryView.as_view()),
    path("patient/<int:pk>/meal-intake/", PatientFoodIntakeSummaryView.as_view()),

    # Update these lines to include the date parameter
    path("patient/<int:pk>/meal-intake/<str:date>", PatientFoodIntakeSummaryByDateView.as_view()),
    path("patient/<int:pk>/meal-intake/<str:date>/", PatientFoodIntakeSummaryByDateView.as_view()),

    # Update these lines to include the date parameter
    path("patient/<int:pk>/meal-intake/<str:date>", PatientFoodIntakeSummaryView.as_view()),
    path("patient/<int:pk>/meal-intake/<str:date>/", PatientFoodIntakeSummaryView.as_view()),

    

    # --- OUR NEW 5090 INGESTION WEBHOOK ---
    path("ingest-5090/", Receive5090PayloadView.as_view()),
    path("ingest-5090", Receive5090PayloadView.as_view()),
    path("ingest-5090/chinese-docs", Receive5090PayloadChineseDocsView.as_view()),
    path("ingest-5090/chinese-docs/", Receive5090PayloadChineseDocsView.as_view()),
]