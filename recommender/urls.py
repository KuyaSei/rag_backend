from django.urls import path
from .views import DailyRecommendationsByPatientView, DailyRecommendationsByPatientAndDateView, MonthlyRecommendationsByPatientView, WeeklyRecommendationsByPatientView

# MonthlyPatientFoodIntakeRecommenderViewMonthlyRecommendationsByDummyPatientView, PatientFoodIntakeRecommenderByDateView, DailyPatientFoodIntakeRecommenderView, , GeneralPatientFoodIntakeRecommenderView, WeeklyRecommendationsByDummyPatientView

# BASE ENDPOINT: /api/recommend/

urlpatterns = [
    # Daily 
    path("nutri-and-food/patient/<int:pk>/daily", DailyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/", DailyRecommendationsByPatientView.as_view()),
 
     # Add this to your urlpatterns:
    # With Date Parameter
    path("nutri-and-food/patient/<int:pk>/daily/<str:date>", DailyRecommendationsByPatientAndDateView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/<str:date>/", DailyRecommendationsByPatientAndDateView.as_view()),

     # Weekly
    path("nutri-and-food/patient/<int:pk>/weekly", WeeklyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/weekly/", WeeklyRecommendationsByPatientView.as_view()),

    # Monthly
    path("nutri-and-food/patient/<int:pk>/monthly", MonthlyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/monthly/", MonthlyRecommendationsByPatientView.as_view()),
    
]

    

