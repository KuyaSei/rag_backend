from urllib import response

from django.shortcuts import render

import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# HPA RAG Services
from rag.services.hpa_retriever_services import build_prompt, build_rag_context, qdrant_search, retrieve_all, retrieve_text

# Patient Docs RAG Services
from rag.services.patient_docs_retriever import format_food_intakes_docs, get_patient_dietary_targets, get_patient_food_intake, get_patient_profile, get_patient_segmented_intake
from rag.services.generator import ask_llm #ask_groq_llm_with_token_limit, ask_ollama_llm, 

# Recommender Services
from recommender.services import calculate_food_item_intake, format_calculated_intakes, format_calculated_intakes_for_response, get_dri_min_max, get_list_of_meals, get_nutrition_remarks, get_nutritional_content_in_json, get_patient_info

# Qdrant Services
from qdrant_client.models import PointStruct, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient




# Daily Recommendations — today or ?date=YYYY-MM-DD query param
class DailyRecommendationsByPatientView(APIView):
    def get(self, request, pk):
        try:
            # 1. INFORMATION RETRIEVAL
            requested_date = request.GET.get('date')
            curdate = requested_date if requested_date else datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

            # Patient Profile
            patient_profile = get_patient_profile(pk)
            if not patient_profile:
                return Response({"detail": f"Patient {pk} not found in the system."}, status=status.HTTP_404_NOT_FOUND)
            patient = patient_profile[0][1]

            # Dietary Targets
            dietary_targets = get_patient_dietary_targets(pk)
            if not dietary_targets:
                return Response({"detail": f"No dietary targets found for patient {pk}."}, status=status.HTTP_404_NOT_FOUND)
            raw_dris = dietary_targets[0][1]
            patient_dris = {
                "calories_kcal": get_dri_min_max(raw_dris["dri_calories"]),
                "protein_g": get_dri_min_max(raw_dris["dri_protein"]),
                "fats_g": get_dri_min_max(raw_dris["dri_fat"]),
                "carbohydrates_g": get_dri_min_max(raw_dris["dri_carbohydrate"]),
                "fiber_g": get_dri_min_max(raw_dris["dri_fiber"]),
            }

            # Food Intakes
            food_intake_results = get_patient_segmented_intake(pk, curdate)
            food_intake_docs = format_food_intakes_docs(food_intake_results)

            # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
            calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)

            lunch_intakes = calculated_intake['by_meal'].get('lunch')
            formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
            lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None

            dinner_intakes = calculated_intake['by_meal'].get('dinner')
            formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
            dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

            lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)

            total_nutri_content = {
                "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
                "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
                "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
                "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
                "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
            }

            nutrition_remarks = {
                "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="protein_g"),
                "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="fats_g"),
                "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="carbohydrates_g"),
                "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="fiber_g"),
                "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="calories_kcal"),
            }

            # 3. DUAL RAG: HPA DOCS RETRIEVAL & LLM GENERATION
            age = patient.get("age")
            sex = patient.get("sex")

            if not age and not sex:
                descriptor = "高齡長者"
            elif age and sex:
                descriptor = f"{age}歲的{sex}"
            elif age:
                descriptor = f"{age}歲"
            else:
                descriptor = sex

            query_1_results = retrieve_all(f"{descriptor}的每日蛋白質攝取建議與需求", top_k=3)
            query_2_results = retrieve_all(f"{descriptor}的每日碳水化合物攝取建議與需求", top_k=3)
            query_3_results = retrieve_all(f"{descriptor}的每日脂質攝取建議與需求", top_k=3)
            query_4_results = retrieve_all(f"{descriptor}的每日熱量(Calories)攝取建議與需求", top_k=3)
            query_5_results = retrieve_all(f"{descriptor}的每日膳食纖維(Fiber)攝取建議與需求", top_k=3)

            meal_names = meal_names_list  # get_list_of_meals()
            meals_text = ", ".join(meal_names)

            prompt = f"""
You are a senior clinical dietitian evaluating a long-term care patient (Room {patient.get('room_number')}, Bed {patient.get('bed_number')}) for the date {curdate}.

You are provided with their actual calculated intake, mathematical remarks, and the official Taiwan HPA Guidelines.

Patient Context:
{get_patient_info(patient)}
Actual Intake: {format_calculated_intakes(calculated_intake.get("aggregated_total", {}))}
Mathematical Remarks: {nutrition_remarks}

HPA Guidelines Context:
Protein: {build_rag_context(query_1_results)}
Carbohydrates: {build_rag_context(query_2_results)}
Lipids/Fats: {build_rag_context(query_3_results)}
Calories: {build_rag_context(query_4_results)}
Fiber: {build_rag_context(query_5_results)}

Available Meals:
{meals_text}

TASK:
You MUST format your response EXACTLY according to the following structure. Do not deviate. Answer in English.

AI Dietary Analysis:
(Write 2-3 sentences synthesizing the Mathematical Remarks with specific thresholds from the HPA Guidelines Context. Explicitly mention the HPA guidelines. Explain what is deficient or excessive clinically.)

Recommended Meals:
(Select exactly 5 meals from the 'Available Meals' list to correct the patient's specific deficits or excesses. Provide a 1-sentence clinical justification for EACH meal.)
1. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
2. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
3. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
4. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
5. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]

Reference Sources (HPA Guidelines):
(List 2-3 specific rules from the 'HPA Guidelines Context' that support your analysis. You MUST extract and print the exact '(Source: [filename])' tag provided in the text. You MUST use the exact numbers provided in the text. Do not invent ranges. Then provide a short description of that rule or explanation )
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])

Rules:
- If there is no food intake record for {curdate}, output ONLY: "Dietary recommendations cannot be provided because no intake was recorded for this day."
- ONLY use meals from the Available Meals list.
"""
            meal_recommendations = ask_llm(prompt)

            # 4. SEND RESPONSE
            response = {
                "response": meal_recommendations,
                "date": curdate,
                "patient": patient,
                "patient_dris": patient_dris,
                "food_intake_docs": food_intake_docs,
                "lunch_items": lunch_items,
                "lunch_nutritional_content": lunch_nutri_content,
                "dinner_items": dinner_items,
                "dinner_nutritional_content": dinner_nutri_content,
                "total_nutritional_content": total_nutri_content,
                "daily_nutrition_remarks": nutrition_remarks,
                "prompt": prompt,
            }

            return Response(response, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"detail": "Error generating response", "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DailyRecommendationsByPatientAndDateView(APIView):    
    def get(self, request, pk, date):
        try:
            # 1. INFORMATION RETRIEVAL
            curdate = date

            # Patient Profile
            patient_profile = get_patient_profile(pk)
            if not patient_profile:
                return Response({"detail": f"Patient {pk} not found in the system."}, status=status.HTTP_404_NOT_FOUND)
            patient = patient_profile[0][1]

            # Dietary Targets (UPDATED TO NEW PREFIXES)
            dietary_targets = get_patient_dietary_targets(pk)
            if not dietary_targets:
                return Response({"detail": f"No dietary targets found for patient {pk}."}, status=status.HTTP_404_NOT_FOUND)
            raw_dris = dietary_targets[0][1]
            
            patient_dris = {
                "calories_kcal": get_dri_min_max(raw_dris["dri_calories"]),
                "protein_g": get_dri_min_max(raw_dris["dri_protein"]),
                "fats_g": get_dri_min_max(raw_dris["dri_fat"]),
                "carbohydrates_g": get_dri_min_max(raw_dris["dri_carbohydrate"]),
                "fiber_g": get_dri_min_max(raw_dris["dri_fiber"]),
            }

            # Food Intakes
            food_intake_results = get_patient_segmented_intake(pk, curdate)
            food_intake_docs = format_food_intakes_docs(food_intake_results)

            # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
            calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
            
            # Format Lunch for LLM (YOUR SAFE HANDLING)
            lunch_intakes = calculated_intake['by_meal'].get('lunch')
            formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
            lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
            # Format Dinner for LLM (YOUR SAFE HANDLING)
            dinner_intakes = calculated_intake['by_meal'].get('dinner')
            formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
            dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

            # Nutritional Content
            lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)

            # Add Total Nutritional Content
            total_nutri_content = {
                "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
                "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
                "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
                "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
                "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
            } 

            # Nutritional Remarks
            nutrition_remarks = {
                "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="protein_g"),
                "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="fats_g"),
                "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="carbohydrates_g"),
                "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="fiber_g"),
                "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="calories_kcal"),
            }        

            # 3. DUAL RAG: HPA DOCS RETRIEVAL & LLM GENERATION
            age = patient.get("age")
            sex = patient.get("sex")
            
            if not age and not sex:
                descriptor = "高齡長者" 
            elif age and sex:
                descriptor = f"{age}歲的{sex}" 
            elif age:
                descriptor = f"{age}歲"
            else:
                descriptor = sex

            query_1_results = retrieve_all(f"{descriptor}的每日蛋白質攝取建議與需求", top_k=3)
            query_2_results = retrieve_all(f"{descriptor}的每日碳水化合物攝取建議與需求", top_k=3)
            query_3_results = retrieve_all(f"{descriptor}的每日脂質攝取建議與需求", top_k=3)
            query_4_results = retrieve_all(f"{descriptor}的每日熱量(Calories)攝取建議與需求", top_k=3)
            query_5_results = retrieve_all(f"{descriptor}的每日膳食纖維(Fiber)攝取建議與需求", top_k=3)

            meal_names = meal_names_list  # get_list_of_meals()
            meals_text = ", ".join(meal_names)
            
            food_intake_context = "\n".join([doc['document'] for doc in food_intake_docs])

            # TRUE DUAL RAG PROMPT
            prompt = f"""
You are a senior clinical dietitian evaluating a long-term care patient (Room {patient.get('room_number')}, Bed {patient.get('bed_number')}) for the date {curdate}.

You are provided with their actual calculated intake, mathematical remarks, and the official Taiwan HPA Guidelines. 

Patient Context:
{get_patient_info(patient)}
Actual Intake: {format_calculated_intakes(calculated_intake.get("aggregated_total", {}))}
Mathematical Remarks: {nutrition_remarks}

HPA Guidelines Context:
Protein: {build_rag_context(query_1_results)}
Carbohydrates: {build_rag_context(query_2_results)}
Lipids/Fats: {build_rag_context(query_3_results)}
Calories: {build_rag_context(query_4_results)}
Fiber: {build_rag_context(query_5_results)}

Available Meals:
{meals_text}

TASK:
You MUST format your response EXACTLY according to the following structure. Do not deviate. Answer in English.

AI Dietary Analysis:
(Write 2-3 sentences synthesizing the Mathematical Remarks with specific thresholds from the HPA Guidelines Context. Explicitly mention the HPA guidelines. Explain what is deficient or excessive clinically.)

Recommended Meals:
(Select exactly 5 meals from the 'Available Meals' list to correct the patient's specific deficits or excesses. Provide a 1-sentence clinical justification for EACH meal.)
1. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
2. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
3. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
4. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
5. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]

Reference Sources (HPA Guidelines):
(List 2-3 specific rules from the 'HPA Guidelines Context' that support your analysis. You MUST extract and print the exact '(Source: [filename])' tag provided in the text. You MUST use the exact numbers provided in the text. Do not invent ranges. Then provide a short description of that rule or explanation )
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])

Rules:
- If there is no food intake record for {curdate}, output ONLY: "Dietary recommendations cannot be provided because no intake was recorded for this day."
- ONLY use meals from the Available Meals list.
"""
            meal_recommendations = ask_llm(prompt)
            

            # 4. SEND RESPONSE
            response = {
                "response": meal_recommendations,
                "date": curdate,
                "patient": patient,
                "patient_dris": patient_dris,
                "food_intake_docs": food_intake_docs,
                "lunch_items": lunch_items,
                "lunch_nutritional_content": lunch_nutri_content,
                "dinner_items": dinner_items,
                "dinner_nutritional_content": dinner_nutri_content,
                "total_nutritional_content": total_nutri_content,
                "daily_nutrition_remarks": nutrition_remarks,
                "prompt": prompt,
            }
            
            return Response(response, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": "Error generating response", "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
       
class WeeklyRecommendationsByPatientView(APIView):
    def get(self, request, pk):
        try:
            # 1. INFORMATION RETRIEVAL
            # Patient Profile
            patient_profile= get_patient_profile(pk)
            if not patient_profile:
                return Response({"detail": f"Patient {pk} not found in the system."}, status=status.HTTP_404_NOT_FOUND)
            patient = patient_profile[0][1]

            # Dietary Targets
            dietary_targets = get_patient_dietary_targets(pk)
            if not dietary_targets:
                return Response({"detail": f"No dietary targets found for patient {pk}."}, status=status.HTTP_404_NOT_FOUND)
            patient_dris = dietary_targets[0][1]
            patient_dris = {
                "calories_kcal": get_dri_min_max(patient_dris["dri_calories"]),
                "protein_g": get_dri_min_max(patient_dris["dri_protein"]),
                "fats_g": get_dri_min_max(patient_dris["dri_fat"]),
                "carbohydrates_g": get_dri_min_max(patient_dris["dri_carbohydrate"]),
                "fiber_g": get_dri_min_max(patient_dris["dri_fiber"]),
            }

            # Food Intakes (Last 7 days)
            # a. Current date
            curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()

            # b. Create a list of dates of the previous 7 days            
            dates_list = [
                (curdate - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(7)
            ]
            
            # c. Get intake docs (PAST 7-DAYS RECORDS)
            weekly_data = {}

            for i, date in enumerate(dates_list):
                intake_results = get_patient_segmented_intake(pk, date)
                food_intake_docs = format_food_intakes_docs(intake_results)

                # d. Create list of docs with day number & dates
                weekly_data[7 - i] = {
                    "date": date,
                    "food_intake_docs": food_intake_docs
                }


            # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
            for day_data in weekly_data.values():
                food_intake_docs = day_data["food_intake_docs"]

                # a. In each meal time (Lunch/Dinner), get intake volume per food class
                calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
                
                    # Format Lunch for LLM (e.g, 40 ml of chicken, 30 ml of broccoli, etc.)
                lunch_intakes = calculated_intake['by_meal'].get('lunch')
                formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
                lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
                    # Format Dinner for LLM
                dinner_intakes = calculated_intake['by_meal'].get('dinner')
                formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
                dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

                
                # b. In each meal time (Lunch/Dinner), get nutritional content of food intakes
                
                    # Lunch Nutritional Content
                lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            
                    # Dinner Nutritional Content
                dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            
                # c. Add Total Nutritional Content for both lunch & dinner 
                total_nutri_content = {
                    "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
                    "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
                    "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
                    "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
                    "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
                } 

                # d. Get nutritional remarks
                nutrition_remarks = {
                    "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "protein_g"),
                    "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fats_g"),
                    "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "carbohydrates_g"),
                    "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fiber_g"),
                    "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "calories_kcal"),
                }

                # e. Append results to each day in weekly_data 
                day_data["lunch_items"] = lunch_items
                day_data["lunch_nutritional_content"] = lunch_nutri_content
                day_data["dinner_items"] = dinner_items
                day_data["dinner_nutritional_content"] = dinner_nutri_content
                day_data["total_nutritional_content"] = total_nutri_content
                day_data["daily_nutrition_remarks"] = nutrition_remarks

            
            # 3. WEEKLY AVERAGE
            # a. Get valid days with food intake records (to only get average of days with records)
            valid_days = [day_data for day_data in weekly_data.values() if day_data["food_intake_docs"]]

            # b. Calculate weekly average for each nutrient based on valid days
            if valid_days:
                weekly_average_nutri_content = {
                    "calories_kcal": sum(day["total_nutritional_content"]["calories_kcal"] for day in valid_days) / len(valid_days),
                    "protein_g": sum(day["total_nutritional_content"]["protein_g"] for day in valid_days) / len(valid_days),
                    "fats_g": sum(day["total_nutritional_content"]["fats_g"] for day in valid_days) / len(valid_days),
                    "carbohydrates_g": sum(day["total_nutritional_content"]["carbohydrates_g"] for day in valid_days) / len(valid_days),
                    "fiber_g": sum(day["total_nutritional_content"]["fiber_g"] for day in valid_days) / len(valid_days),
                }
            else:
                # c. If no valid days (i.e., no food intake records), set weekly average to 0 or None
                weekly_average_nutri_content = {
                    "calories_kcal": 0,
                    "protein_g": 0,
                    "fats_g": 0,
                    "carbohydrates_g": 0,
                    "fiber_g": 0,
                }

                # d. GET NUTRITIONAL REMARKS BASED ON RECOMMENDED INTAKE AND WEEKLY AVERAGE INTAKE 
            weekly_nutrition_remarks = {
                "protein_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "protein_g"),
                "fats_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fats_g"),
                "carbohydrates_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "carbohydrates_g"),
                "fiber_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fiber_g"),
                "calories_kcal": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "calories_kcal"),
            }


            # 4. DUAL RAG: HPA DOCS RETRIEVAL + LLM GENERATION
            age = patient.get("age")
            sex = patient.get("sex")
            if not age and not sex:
                descriptor = "高齡長者"
            elif age and sex:
                descriptor = f"{age}歲的{sex}"
            elif age:
                descriptor = f"{age}歲"
            else:
                descriptor = sex

            query_1_results = retrieve_all(f"{descriptor}的每日蛋白質攝取建議與需求", top_k=3)
            query_2_results = retrieve_all(f"{descriptor}的每日碳水化合物攝取建議與需求", top_k=3)
            query_3_results = retrieve_all(f"{descriptor}的每日脂質攝取建議與需求", top_k=3)
            query_4_results = retrieve_all(f"{descriptor}的每日熱量(Calories)攝取建議與需求", top_k=3)
            query_5_results = retrieve_all(f"{descriptor}的每日膳食纖維(Fiber)攝取建議與需求", top_k=3)

            meal_names = meal_names_list  # get_list_of_meals()
            meals_text = ", ".join(meal_names)

            prompt = f"""
You are a senior clinical dietitian evaluating a long-term care patient (Room {patient.get('room_number')}, Bed {patient.get('bed_number')}) for the past 7 days.

You are provided with their weekly average calculated intake, mathematical remarks, and the official Taiwan HPA Guidelines.

Patient Context:
{get_patient_info(patient)}
Weekly Average Intake: {weekly_average_nutri_content}
Mathematical Remarks: {weekly_nutrition_remarks}

HPA Guidelines Context:
Protein: {build_rag_context(query_1_results)}
Carbohydrates: {build_rag_context(query_2_results)}
Lipids/Fats: {build_rag_context(query_3_results)}
Calories: {build_rag_context(query_4_results)}
Fiber: {build_rag_context(query_5_results)}

Available Meals:
{meals_text}

TASK:
You MUST format your response EXACTLY according to the following structure. Do not deviate. Answer in English.

AI Dietary Analysis:
(Write 2-3 sentences synthesizing the Mathematical Remarks with specific thresholds from the HPA Guidelines Context. Explicitly mention the HPA guidelines. Explain what is deficient or excessive clinically based on the weekly average.)

Recommended Meals:
(Select exactly 5 meals from the 'Available Meals' list to correct the patient's specific deficits or excesses. Provide a 1-sentence clinical justification for EACH meal.)
1. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
2. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
3. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
4. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
5. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]

Reference Sources (HPA Guidelines):
(List 2-3 specific rules from the 'HPA Guidelines Context' that support your analysis. You MUST extract and print the exact '(Source: [filename])' tag provided in the text. You MUST use the exact numbers provided in the text. Do not invent ranges.)
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])

Rules:
- If there are no food intake records for the past 7 days, output ONLY: "Dietary recommendations cannot be provided because no intake was recorded for this period."
- ONLY use meals from the Available Meals list.
"""
            meal_recos = ask_llm(prompt)

            
            # 5. FORMAT weekly_data FOR RESPONSE
            for day_data in weekly_data.values():
                day_data.pop("food_intake_docs", None)


            # 6. SEND RESPONSE
            response = {
                    "response": meal_recos,
                    "dates_list": dates_list,
                    "patient_dris": patient_dris,
                    "weekly_data": weekly_data,
                    "weekly_average_nutritional_content": weekly_average_nutri_content,
                    "weekly_nutrition_remarks": weekly_nutrition_remarks,
                }
            
            return Response(
                response,
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
        
class MonthlyRecommendationsByPatientView(APIView):
    def get(self, request, pk):
        try:
            # 1. INFORMATION RETRIEVAL
            # Patient Profile
            patient_profile= get_patient_profile(pk)
            if not patient_profile:
                return Response({"detail": f"Patient {pk} not found in the system."}, status=status.HTTP_404_NOT_FOUND)
            patient = patient_profile[0][1]

            # Dietary Targets
            dietary_targets = get_patient_dietary_targets(pk)
            if not dietary_targets:
                return Response({"detail": f"No dietary targets found for patient {pk}."}, status=status.HTTP_404_NOT_FOUND)
            patient_dris = dietary_targets[0][1]
            patient_dris = {
                "calories_kcal": get_dri_min_max(patient_dris["dri_calories"]),
                "protein_g": get_dri_min_max(patient_dris["dri_protein"]),
                "fats_g": get_dri_min_max(patient_dris["dri_fat"]),
                "carbohydrates_g": get_dri_min_max(patient_dris["dri_carbohydrate"]),
                "fiber_g": get_dri_min_max(patient_dris["dri_fiber"]),
            }

            
            # Food Intakes (Last 28 days)
            TOTAL_DAYS = 28
            # a. Current date
            curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()

            # b. Create a list of dates of the previous 28 days            
            dates_list = [
                (curdate - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(TOTAL_DAYS)
            ]
            
            # c. Get intake docs (PAST 28-DAYS RECORDS)
            monthly_data = {}

            for i, date in enumerate(dates_list):
                intake_results = get_patient_segmented_intake(pk, date)
                food_intake_docs = format_food_intakes_docs(intake_results)

                # d. Create list of docs with day number & dates
                monthly_data[TOTAL_DAYS - i]= {
                    "date": date,
                    "food_intake_docs": food_intake_docs
                
                }


            # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
            for day_data in monthly_data.values():
                food_intake_docs = day_data["food_intake_docs"]

                # a. In each meal time (Lunch/Dinner), get intake volume per food class
                calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
                
                    # Format Lunch for LLM (e.g, 40 ml of chicken, 30 ml of broccoli, etc.)
                lunch_intakes = calculated_intake['by_meal'].get('lunch')
                formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
                lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
                    # Format Dinner for LLM
                dinner_intakes = calculated_intake['by_meal'].get('dinner')
                formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
                dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

                
                # b. In each meal time (Lunch/Dinner), get nutritional content of food intakes
                
                    # Lunch Nutritional Content
                lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            
                    # Dinner Nutritional Content
                dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            
                # c. Add Total Nutritional Content for both lunch & dinner 
                total_nutri_content = {
                    "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
                    "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
                    "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
                    "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
                    "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
                } 

                # d. Get nutritional remarks
                nutrition_remarks = {
                    "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "protein_g"),
                    "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fats_g"),
                    "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "carbohydrates_g"),
                    "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fiber_g"),
                    "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "calories_kcal"),
                }

                # e. Append results to each day in weekly_data 
                day_data["lunch_items"] = lunch_items
                day_data["lunch_nutritional_content"] = lunch_nutri_content
                day_data["dinner_items"] = dinner_items
                day_data["dinner_nutritional_content"] = dinner_nutri_content
                day_data["total_nutritional_content"] = total_nutri_content
                day_data["daily_nutrition_remarks"] = nutrition_remarks

            
            # 3. WEEKLY AVERAGE (4 weeks)
            # a. Split 28 days into 4 weeks
            # Week 1 → days 1–7 (oldest)
            # Week 4 → days 22–28 (latest)
            monthly_list = list(monthly_data.values())
            # Reverse so oldest → latest
            monthly_list.reverse()
            weeks = [
                monthly_list[i:i+7]
                for i in range(0, len(monthly_list), 7)
            ]
            
            weekly_data = {}

            for i, week in enumerate(weeks, start=1):
                # b. Get valid days with food intake records (to only get average of days with records)
                valid_days = [day for day in week if day.get("food_intake_docs")]
            
                # c. Calculate weekly average for each nutrient based on valid days
                num = len(valid_days)
                if valid_days:
                    weekly_average_nutri_content = {
                        "calories_kcal": sum(day["total_nutritional_content"]["calories_kcal"] for day in valid_days) / num,
                        "protein_g": sum(day["total_nutritional_content"]["protein_g"] for day in valid_days) / num,
                        "fats_g": sum(day["total_nutritional_content"]["fats_g"] for day in valid_days) / num,
                        "carbohydrates_g": sum(day["total_nutritional_content"]["carbohydrates_g"] for day in valid_days) / num,
                        "fiber_g": sum(day["total_nutritional_content"]["fiber_g"] for day in valid_days) / num,
                    }
                else:
                    # d. If no valid days (i.e., no food intake records), set weekly average to 0 or None
                    weekly_average_nutri_content = {
                        "calories_kcal": 0,
                        "protein_g": 0,
                        "fats_g": 0,
                        "carbohydrates_g": 0,
                        "fiber_g": 0,
                    }
            
                    # d. GET NUTRITIONAL REMARKS BASED ON RECOMMENDED INTAKE AND WEEKLY AVERAGE INTAKE 
                weekly_nutrition_remarks = {
                    "protein_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "protein_g"),
                    "fats_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fats_g"),
                    "carbohydrates_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "carbohydrates_g"),
                    "fiber_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fiber_g"),
                    "calories_kcal": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "calories_kcal"),
                }

                weekly_data[i] = {
                    "weekly_average_nutritional_content": weekly_average_nutri_content,
                    "weekly_nutrition_remarks": weekly_nutrition_remarks
                }

            # 4. MONTHLY AVERAGE
            # a. Get valid weeks with food intake records (to only get average of weeks with records)
            valid_weeks = [    
                week for week in weekly_data.values()
                if week["weekly_average_nutritional_content"]["protein_g"] != 0
            ]

            # b. Calculate monthly average for each nutrient based on valid days
            num = len(valid_weeks)
            if valid_weeks:
                monthly_average_nutri_content = {
                    "calories_kcal": sum(week["weekly_average_nutritional_content"]["calories_kcal"] for week in valid_weeks) / num,
                    "protein_g": sum(week["weekly_average_nutritional_content"]["protein_g"] for week in valid_weeks) / num,
                    "fats_g": sum(week["weekly_average_nutritional_content"]["fats_g"] for week in valid_weeks) / num,
                    "carbohydrates_g": sum(week["weekly_average_nutritional_content"]["carbohydrates_g"] for week in valid_weeks) / num,
                    "fiber_g": sum(week["weekly_average_nutritional_content"]["fiber_g"] for week in valid_weeks) / num,
                }
            else:
                # c. If no valid weeks (i.e., no intakes & nutrients), set monthly average to 0 or None
                monthly_average_nutri_content = {
                    "calories_kcal": 0,
                    "protein_g": 0,
                    "fats_g": 0,
                    "carbohydrates_g": 0,
                    "fiber_g": 0,
                }

                # d. GET NUTRITIONAL REMARKS BASED ON RECOMMENDED INTAKE AND MONTHLY AVERAGE INTAKE 
            monthly_nutrition_remarks = {
                "protein_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "protein_g"),
                "fats_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "fats_g"),
                "carbohydrates_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "carbohydrates_g"),
                "fiber_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "fiber_g"),
                "calories_kcal": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "calories_kcal"),
            }


            # 4. DUAL RAG: HPA DOCS RETRIEVAL + LLM GENERATION
            age = patient.get("age")
            sex = patient.get("sex")
            if not age and not sex:
                descriptor = "高齡長者"
            elif age and sex:
                descriptor = f"{age}歲的{sex}"
            elif age:
                descriptor = f"{age}歲"
            else:
                descriptor = sex

            query_1_results = retrieve_all(f"{descriptor}的每日蛋白質攝取建議與需求", top_k=3)
            query_2_results = retrieve_all(f"{descriptor}的每日碳水化合物攝取建議與需求", top_k=3)
            query_3_results = retrieve_all(f"{descriptor}的每日脂質攝取建議與需求", top_k=3)
            query_4_results = retrieve_all(f"{descriptor}的每日熱量(Calories)攝取建議與需求", top_k=3)
            query_5_results = retrieve_all(f"{descriptor}的每日膳食纖維(Fiber)攝取建議與需求", top_k=3)

            meal_names = meal_names_list  # get_list_of_meals()
            meals_text = ", ".join(meal_names)

            prompt = f"""
You are a senior clinical dietitian evaluating a long-term care patient (Room {patient.get('room_number')}, Bed {patient.get('bed_number')}) for the past 28 days.

You are provided with their monthly average calculated intake, mathematical remarks, and the official Taiwan HPA Guidelines.

Patient Context:
{get_patient_info(patient)}
Monthly Average Intake: {monthly_average_nutri_content}
Mathematical Remarks: {monthly_nutrition_remarks}

HPA Guidelines Context:
Protein: {build_rag_context(query_1_results)}
Carbohydrates: {build_rag_context(query_2_results)}
Lipids/Fats: {build_rag_context(query_3_results)}
Calories: {build_rag_context(query_4_results)}
Fiber: {build_rag_context(query_5_results)}

Available Meals:
{meals_text}

TASK:
You MUST format your response EXACTLY according to the following structure. Do not deviate. Answer in English.

AI Dietary Analysis:
(Write 2-3 sentences synthesizing the Mathematical Remarks with specific thresholds from the HPA Guidelines Context. Explicitly mention the HPA guidelines. Explain what is deficient or excessive clinically based on the monthly average.)

Recommended Meals:
(Select exactly 5 meals from the 'Available Meals' list to correct the patient's specific deficits or excesses. Provide a 1-sentence clinical justification for EACH meal.)
1. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
2. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
3. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
4. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]
5. [Meal Name in Chinese] ([Meal Name translated to English]) - [Specific clinical reason based on macros/micros]

Reference Sources (HPA Guidelines):
(List 2-3 specific rules from the 'HPA Guidelines Context' that support your analysis. You MUST extract and print the exact '(Source: [filename])' tag provided in the text. You MUST use the exact numbers provided in the text. Do not invent ranges.)
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])
- [Specific Rule with EXACT numbers from context] - (Source: [Exact Document Name.pdf])

Rules:
- If there are no food intake records for the past 28 days, output ONLY: "Dietary recommendations cannot be provided because no intake was recorded for this period."
- ONLY use meals from the Available Meals list.
"""
            meal_recos = ask_llm(prompt)

            
            # 5. FORMAT monthly_data FOR RESPONSE
            for day_data in monthly_data.values():
                day_data.pop("food_intake_docs", None)


            # 6. SEND RESPONSE
            response = {
                "response": meal_recos,
                "dates_list": dates_list,
                "patient_dris": patient_dris,
                "monthly_data": monthly_data,
                "weekly_data": weekly_data,
                "monthly_average_nutritional_content": monthly_average_nutri_content,
                "monthly_nutrition_remarks": monthly_nutrition_remarks,
            }

            return Response(
                response,
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



# From actual db
meal_names_list = ['滷肉排', '馬鈴薯炒肉末', '蒜醬麵腸', '炒時蔬', '筍絲豆皮湯', '滷油干魚', '滷豆支', '四季豆炒香腸', '炒時蔬', '紫菜蛋花湯', '什錦烏龍麵', '炒時蔬', '大白菜豆皮湯', '高麗菜粥', '馬鈴薯燉肉', '芋頭蛋黃球', '滷蘿蔔', '炒時蔬', '味噌蛋花湯', '滷雞排', '炒冬粉', '薑絲海帶根', '炒時蔬', '玉米湯', '洋蔥炒豬柳', '燴咖哩', '炒時蔬', '海帶芽湯', '銀斑魚', '沙茶素腰花', '玉米炒蛋', '炒時蔬', '高麗菜湯', '滷雞腿', '紅蘿蔔滷貢丸', '蒜醬百頁豆腐', '炒時蔬', '紫菜蛋花湯', '紅燒獅子頭', '蘿蔔滷豆輪', '馬鈴薯炒蛋', '炒時蔬', '玉米湯', '滷肉排', '海苔丸', '醬拌豆干', '炒時蔬', '冬菜豆芽湯', '碗粿', '筍絲豆皮湯', '皮蛋鹹粥', '日式豬排', '滷麵筋', '薑絲炒木耳', '炒時蔬', '海帶芽湯', '古早味炒麵', '蘿蔔湯', '絲瓜鹹粥', '滷雞排', '紅蘿蔔炒蛋', '馬鈴薯炒肉末', '炒時蔬', '玉米湯', '滷油干魚', '肉末滷油豆腐', '茄汁炒蛋', '炒時蔬', '筍絲豆皮湯', '香腸', '炒冬粉', '馬鈴薯炒肉末', '炒時蔬', '紫菜蛋花湯', '三杯里肌', '洋蔥炒甜不辣', '紅蘿蔔炒蛋', '炒時蔬', '冬菜豆芽湯', '滷雞腿', '蘿蔔滷豆輪', '炒時蔬', '味噌蛋花湯', '什錦米粉', '炒時蔬', '大白菜豆皮湯', '芋頭鹹粥', '滷銀斑魚', '滷筍絲', '蒜醬百頁豆腐', '炒時蔬', '海帶芽湯', '沙茶腿排', '三杯麵腸', '玉米炒蛋', '炒時蔬', '鳳梨苦瓜湯', '紅燒獅子頭', '滷海帶結', '炒時蔬', '玉米湯', '馬鈴薯燉肉', '滷蘿蔔', '炒冬粉', '炒時蔬', '筍絲豆皮湯', '炸無骨雞排', '燴咖哩', '紅蘿蔔炒蛋', '炒時蔬', '紫菜湯', '米糕', '鳳梨苦瓜湯', '高麗菜鹹粥', '紅燒里肌', '醬拌豆干', '炒時蔬', '玉米湯', '雞肉飯', '筍絲豆皮湯', '絲瓜鹹粥', '滷銀斑魚', '滷豆支', '沙茶玉米炒肉末', '炒時蔬', '大白菜豆皮湯']








