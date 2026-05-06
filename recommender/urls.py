from django.urls import path
from .views import (
    DailyRecommendationsByPatientView,
    DailyRecommendationsByPatientAndDateView,
    WeeklyRecommendationsByPatientView,
    MonthlyRecommendationsByPatientView,
    WeeklyTrendByPatientView,
    MonthlyTrendByPatientView,
)

# BASE ENDPOINT: /api/recommend/

urlpatterns = [
    # Daily
    path("nutri-and-food/patient/<int:pk>/daily", DailyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/", DailyRecommendationsByPatientView.as_view()),

    # Daily with specific date
    path("nutri-and-food/patient/<int:pk>/daily/<str:date>", DailyRecommendationsByPatientAndDateView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/<str:date>/", DailyRecommendationsByPatientAndDateView.as_view()),

    # Weekly recommendation (with LLM)
    path("nutri-and-food/patient/<int:pk>/weekly", WeeklyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/weekly/", WeeklyRecommendationsByPatientView.as_view()),

    # Monthly recommendation (with LLM)
    path("nutri-and-food/patient/<int:pk>/monthly", MonthlyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/monthly/", MonthlyRecommendationsByPatientView.as_view()),

    # Weekly trend data (no LLM — for chart)
    path("nutri-and-food/patient/<int:pk>/weekly/trend", WeeklyTrendByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/weekly/trend/", WeeklyTrendByPatientView.as_view()),

    # Monthly trend data (no LLM — for chart)
    path("nutri-and-food/patient/<int:pk>/monthly/trend", MonthlyTrendByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/monthly/trend/", MonthlyTrendByPatientView.as_view()),
]

    

