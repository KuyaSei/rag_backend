from django.urls import path
from .views import (
    DailyRecommendationsByPatientView,
    DailyRecommendationsByPatientAndDateView,
    WeeklyRecommendationsByPatientView,
    MonthlyRecommendationsByPatientView,
    WeeklyTrendByPatientView,
    MonthlyTrendByPatientView,
    ClusteredDailyRecommendationsByPatientView,
    ClusteredWeeklyRecommendationsByPatientView,
    ClusteredMonthlyRecommendationsByPatientView,
    PatientMacroSummaryView,
    KnnJustifiedRecommendationView,
    KnnJustifiedWeeklyView,
    KnnJustifiedMonthlyView,
)

# BASE ENDPOINT: /api/recommend/

urlpatterns = [
    # Daily
    path("nutri-and-food/patient/<int:pk>/daily", DailyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/", DailyRecommendationsByPatientView.as_view()),

    # Daily with K-Means cluster-filtered meals (must come before <str:date> pattern)
    path("nutri-and-food/patient/<int:pk>/daily/clustered", ClusteredDailyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/clustered/", ClusteredDailyRecommendationsByPatientView.as_view()),

    # Daily with specific date
    path("nutri-and-food/patient/<int:pk>/daily/<str:date>", DailyRecommendationsByPatientAndDateView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/<str:date>/", DailyRecommendationsByPatientAndDateView.as_view()),

    # Weekly recommendation (with LLM)
    path("nutri-and-food/patient/<int:pk>/weekly", WeeklyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/weekly/", WeeklyRecommendationsByPatientView.as_view()),

    # Weekly recommendation with K-Means cluster-filtered meals
    path("nutri-and-food/patient/<int:pk>/weekly/clustered", ClusteredWeeklyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/weekly/clustered/", ClusteredWeeklyRecommendationsByPatientView.as_view()),

    # Monthly recommendation (with LLM)
    path("nutri-and-food/patient/<int:pk>/monthly", MonthlyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/monthly/", MonthlyRecommendationsByPatientView.as_view()),

    # Monthly recommendation with K-Means cluster-filtered meals
    path("nutri-and-food/patient/<int:pk>/monthly/clustered", ClusteredMonthlyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/monthly/clustered/", ClusteredMonthlyRecommendationsByPatientView.as_view()),

    # Weekly trend data (no LLM — for chart)
    path("nutri-and-food/patient/<int:pk>/weekly/trend", WeeklyTrendByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/weekly/trend/", WeeklyTrendByPatientView.as_view()),

    # Monthly trend data (no LLM — for chart)
    path("nutri-and-food/patient/<int:pk>/monthly/trend", MonthlyTrendByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/monthly/trend/", MonthlyTrendByPatientView.as_view()),

    # Lightweight macro summary for KNN (no LLM — daily/weekly/monthly via ?period=&date=)
    path("nutri-and-food/patient/<int:pk>/macro-summary", PatientMacroSummaryView.as_view()),
    path("nutri-and-food/patient/<int:pk>/macro-summary/", PatientMacroSummaryView.as_view()),

    # KNN (Alex) + LLM clinical justification — calls Alex's KNN then layers HPA RAG on top
    path("nutri-and-food/patient/<int:pk>/knn-justified", KnnJustifiedRecommendationView.as_view()),
    path("nutri-and-food/patient/<int:pk>/knn-justified/", KnnJustifiedRecommendationView.as_view()),

    # KNN justified — weekly (7-day nutrition table + KNN period=week)
    path("nutri-and-food/patient/<int:pk>/knn-justified/weekly", KnnJustifiedWeeklyView.as_view()),
    path("nutri-and-food/patient/<int:pk>/knn-justified/weekly/", KnnJustifiedWeeklyView.as_view()),

    # KNN justified — monthly (28-day nutrition table + KNN period=month)
    path("nutri-and-food/patient/<int:pk>/knn-justified/monthly", KnnJustifiedMonthlyView.as_view()),
    path("nutri-and-food/patient/<int:pk>/knn-justified/monthly/", KnnJustifiedMonthlyView.as_view()),
]

    

