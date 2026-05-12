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
]

    

