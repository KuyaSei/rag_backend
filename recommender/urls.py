from django.urls import path
from .views import DailyRecommendationsByPatientView, WeeklyPatientFoodIntakeRecommenderView, DailyRecommendationsByPatientAndDateView

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
    
]

   # path("test/patient/<int:pk>/daily/", GeneralPatientFoodIntakeRecommenderView.as_view()),

    # path("nutri-and-food/patient/dummy/daily", DailyRecommendationsByDummyPatientView.as_view()),
    # path("nutri-and-food/patient/dummy/daily/", DailyRecommendationsByDummyPatientView.as_view()),

    # path("nutri-and-food/patient/dummy/weekly", WeeklyRecommendationsByDummyPatientView.as_view()),
    # path("nutri-and-food/patient/dummy/weekly/", WeeklyRecommendationsByDummyPatientView.as_view()),

    # path("nutri-and-food/patient/dummy/monthly", MonthlyRecommendationsByDummyPatientView.as_view()),
    # path("nutri-and-food/patient/dummy/monthly/", MonthlyRecommendationsByDummyPatientView.as_view()),


    # [PREVIOUS]
    # Patient Daily food intake recommendations
    # path("patient/<int:pk>/daily", DailyPatientFoodIntakeRecommenderView.as_view()),
    # path("patient/<int:pk>/daily/", DailyPatientFoodIntakeRecommenderView.as_view()),

    # Patient food intake recommendations By Date
    # path("patient/<int:pk>/daily/<str:date>", PatientFoodIntakeRecommenderByDateView.as_view()),
    # path("patient/<int:pk>/daily/<str:date>/", PatientFoodIntakeRecommenderByDateView.as_view()),`

    # Patient Weekly food intake recommendations
    # path("patient/<int:pk>/weekly", WeeklyPatientFoodIntakeRecommenderView.as_view()),
    # path("patient/<int:pk>/weekly/", WeeklyPatientFoodIntakeRecommenderView.as_view()),

    # Patient Monthly food intake recommendations
    # path("patient/<int:pk>/monthly", MonthlyPatientFoodIntakeRecommenderView.as_view()),
    # path("patient/<int:pk>/monthly/", MonthlyPatientFoodIntakeRecommenderView.as_view()),
    

