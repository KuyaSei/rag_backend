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
from recommender.services import calculate_food_item_intake, create_monthly_food_intake_context, format_calculated_intakes, format_calculated_intakes_for_response, get_dates_in_current_month, get_food_intake_results_in_curmonth, get_list_of_meals,  get_nutrition_remarks, get_nutritional_content_in_json, get_patient_info

# Qdrant Services
from qdrant_client.models import PointStruct, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient

# EMYEEEEL'S FOOD INTAKE BACKEND URL
FOOD_INTAKE_BACKEND_URL = "https://q30gkzkn-8000.asse.devtunnels.ms/api/"

# Obsolete - Not used currently on the flow on the analysis page

# class GeneralPatientFoodIntakeRecommenderView(APIView):    
#     def get(self, request, pk):
#         try:
#             # 1. INFORMATION RETRIEVAL
#             # 1.1 Get patient info & patient profile doc
#             patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()   
#             patient_profile_doc= get_patient_info(patient)

#             return Response(
#                 {
#                     "patient_profile_doc": patient_profile_doc,
#                 },
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )

# OKAY - LIVE DATA FOR THE ANALYSIS PAGE (Daily Dropdown)             
class DailyRecommendationsByPatientView(APIView):    
    def get(self, request, pk):
        try:
            # 1. INFORMATION RETRIEVAL
            # Current Day
            
            # Check if frontend passed a specific date parameter (e.g., ?date=YYYY-MM-DD)
            requested_date = request.GET.get('date')
            
            if requested_date:
                curdate = requested_date # Use the user's selected date
            else:
                curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d") # Default to today

            # Patient Profile
            patient_profile= get_patient_profile(pk)
            patient = patient_profile[0][1]

            # Dietary Targets
            dietary_targets = get_patient_dietary_targets(pk)
            patient_dris = dietary_targets[0][1]
            patient_dris = {
                "calories_kcal": patient_dris["total_calories_kcal"],
                "protein_g": patient_dris["total_protein_g"],
                "fats_g": patient_dris["total_fats_g"],
                "carbohydrates_g": patient_dris["total_carbohydrates_g"],
                "fiber_g": patient_dris["total_fiber_g"],
            }

            # Food Intakes (Current date)
            food_intake_results = get_patient_segmented_intake(pk, curdate)
            food_intake_docs = format_food_intakes_docs(food_intake_results)

            # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
            # a. In each meal time (Lunch/Dinner), get food intake volume per food class
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
            print(f"Lunch Nutritional Content: {lunch_nutri_content}")

                # Dinner Nutritional Content
            dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            print(f"Dinner Nutritional Content: {dinner_nutri_content}")

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


            # 3. HPA DOCS RETRIEVAL & LLM GENERATION (TRUE DUAL RAG)
            
            # Extract demographic info for semantic search
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

            # Fetch ALL 5 Macros
            query_1_results = retrieve_all(f"{descriptor}的每日蛋白質攝取建議與需求", top_k=3)
            query_2_results = retrieve_all(f"{descriptor}的每日碳水化合物攝取建議與需求", top_k=3)
            query_3_results = retrieve_all(f"{descriptor}的每日脂質攝取建議與需求", top_k=3)
            query_4_results = retrieve_all(f"{descriptor}的每日熱量(Calories)攝取建議與需求", top_k=3)
            query_5_results = retrieve_all(f"{descriptor}的每日膳食纖維(Fiber)攝取建議與需求", top_k=3)

            meal_names = get_list_of_meals()
            meals_text = ", ".join(meal_names)
            
            # Build Food Intake Context String
            food_intake_context = "\n".join([doc['document'] for doc in food_intake_docs])

            # CREATE PROMPT (LLM Instructions in English)
            prompt = f"""
Based on the following information, evaluate the patient's diet and provide dietary recommendations to the long-term care patient on room {patient.get('room_number')} bed {patient.get('bed_number')} based on food intake record for the day {curdate}.

Task:
1. Evaluate the patient's actual intake of {format_calculated_intakes(calculated_intake.get("aggregated_total", {}))} against the retrieved HPA Guidelines for all macros (Calories, Protein, Carbohydrates, Fats, Fiber).
2. Note any deficiencies or excesses based on the math remarks: {nutrition_remarks}.
3. Based on the evaluation, recommend 5 suitable meals from the provided list to help balance the patient's diet.

Context:
Patient Information:
{get_patient_info(patient)}

Food intakes for {curdate}:
{food_intake_context}

Available Meals:
{meals_text}

HPA Guidelines (Translate and apply these rules):
Protein: {build_rag_context(query_1_results)}
Carbohydrates: {build_rag_context(query_2_results)}
Lipids/Fats: {build_rag_context(query_3_results)}
Calories: {build_rag_context(query_4_results)}
Fiber: {build_rag_context(query_5_results)}

Rules:
- If there is no food intake record for {curdate}, refrain from answering queries and politely explain that dietary recommendations cannot be provided because no intake was recorded for that day.
- ONLY recommend meals that explicitly exist in the 'Available Meals' list.
- Keep the response concise, professional, and under 200 words.
- ONLY answer in English. Do not use Chinese characters in the final output.
"""
            # Generate response directly bypassing services.py
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

# OBSOLETE - Original version before adding date parameter and dual RAG (HPA + Patient Docs)
class DailyRecommendationsByPatientAndDateView(APIView):    
    def get(self, request, pk, date): # <-- Notice it accepts 'date' now
        try:
            # 1. INFORMATION RETRIEVAL
            # INSTEAD of datetime.now(), we use the date from the URL
            curdate = date

            # Patient Profile
            patient_profile= get_patient_profile(pk)
            patient = patient_profile[0][1]

            # Dietary Targets
            dietary_targets = get_patient_dietary_targets(pk)
            patient_dris = dietary_targets[0][1]
            patient_dris = {
                "calories_kcal": get_dri_min_max(patient_dris["dri_calories"]),
                "protein_g": get_dri_min_max(patient_dris["dri_protein"]),
                "fats_g": get_dri_min_max(patient_dris["dri_fat"]),
                "carbohydrates_g": get_dri_min_max(patient_dris["dri_carbohydrate"]),
                "fiber_g": get_dri_min_max(patient_dris["dri_fiber"]),
            }

            # Food Intakes (Using the date from the URL)
            food_intake_results = get_patient_segmented_intake(pk, curdate)
            food_intake_docs = format_food_intakes_docs(food_intake_results)

            # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
            # a. In each meal time (Lunch/Dinner), get food intake volume per food class
            calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
            
            # Format Lunch for LLM
            lunch_intakes = calculated_intake['by_meal'].get('lunch')
            formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
            lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
            # Format Dinner for LLM
            dinner_intakes = calculated_intake['by_meal'].get('dinner')
            formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
            dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

            # b. In each meal time (Lunch/Dinner), get nutritional content of food intakes
            lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
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

            # 3. CREATE PROMPT & GENERATE RESPONSE WITH LLM
            meal_names = meal_names_list 
            meal_recommendations = get_daily_meal_recommendations(meal_names, nutrition_remarks)

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

# class DailyRecommendationsByPatientAndDateView(APIView):    
#     def get(self, request, pk, date):
#         try:
#             # 1. INFORMATION RETRIEVAL
#             # Using the date passed from the URL parameter
#             curdate = date

#             # 1.1 Patient Profile (Fetch the patient profile metadatas)
#             patient_profile = get_patient_profile(pk)
#             patient = patient_profile[0][1]

#             # 1.2 Dietary Targets (Mapped to the dri prefixes)
#             dietary_targets = get_patient_dietary_targets(pk)
#             raw_dris = dietary_targets[0][1]
            
#             # Map raw static targets into Min/Max range objects for the frontend table
#             patient_dris = {
#                 "calories_kcal": get_dri_min_max(raw_dris["total_calories_kcal"]),
#                 "protein_g": get_dri_min_max(raw_dris["total_protein_g"]),
#                 "fats_g": get_dri_min_max(raw_dris["total_fats_g"]),
#                 "carbohydrates_g": get_dri_min_max(raw_dris["total_carbohydrates_g"]),
#                 "fiber_g": get_dri_min_max(raw_dris["total_fiber_g"]),
#             }

#             # 1.3 Food Intakes (Fetch segmented documents for current date)
#             food_intake_results = get_patient_segmented_intake(pk, curdate)
#             food_intake_docs = format_food_intakes_docs(food_intake_results)

#             # 2. FOOD VOLUME -> NUTRITIONAL CONTENT (Your core math contribution)
#             # a. Calculate volume per class by subtracting 'After' from 'Before'[cite: 16]
#             calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
            
#             # Format meal breakdown for JSON response[cite: 16]
#             lunch_intakes = calculated_intake['by_meal'].get('lunch')
#             lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
#             dinner_intakes = calculated_intake['by_meal'].get('dinner')
#             dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

#             # b. Calculate nutritional content of daily intakes (Daily pass aggregation)
#             # Lunch Nutritional Content
#             lunch_nutri_content = get_nutritional_content_in_json(format_calculated_intakes(lunch_intakes))
#             # Dinner Nutritional Content
#             dinner_nutri_content = get_nutritional_content_in_json(format_calculated_intakes(dinner_intakes))

#             # c. Add Total Nutritional Content for both lunch & dinner 
#             total_nutri_content = {
#                 "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
#                 "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
#                 "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
#                 "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
#                 "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
#             } 

#             # d. Generate Remarks using the Min/Max Ranges and calculated totals
#             nutrition_remarks = {
#                 "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="protein_g"),
#                 "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="fats_g"),
#                 "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="carbohydrates_g"),
#                 "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="fiber_g"),
#                 "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient="calories_kcal"),
#             }        

#             # 3. DUAL RAG: HPA DOCS RETRIEVAL & LLM GENERATION
#             # Semantic search against Taiwan HPA Guidelines based on demographic
#             age = patient.get("age")
#             sex = patient.get("sex")
            
#             if not age and not sex:
#                 descriptor = "高齡長者" 
#             elif age and sex:
#                 descriptor = f"{age}歲的{sex}" 
#             elif age:
#                 descriptor = f"{age}歲"
#             else:
#                 descriptor = sex

#             # Retrieve ALL 5 specific macro contexts from HPA Vector DB
#             query_1_results = retrieve_all(f"{descriptor}的每日蛋白質攝取建議與需求", top_k=3)
#             query_2_results = retrieve_all(f"{descriptor}的每日碳水化合物攝取建議與需求", top_k=3)
#             query_3_results = retrieve_all(f"{descriptor}的每日脂質攝取建議與需求", top_k=3)
#             query_4_results = retrieve_all(f"{descriptor}的每日熱量(Calories)攝取建議與需求", top_k=3)
#             query_5_results = retrieve_all(f"{descriptor}的每日膳食纖維(Fiber)攝取建議與需求", top_k=3)

#             meal_names = get_list_of_meals()
#             meals_text = ", ".join(meal_names)

#             food_intake_context = "\n".join([doc['document'] for doc in food_intake_docs])

#             # CREATE PROFESSIONAL PROMPT (English output instructions)
#             prompt = f"""
# Based on the following information, evaluate the patient's diet and provide dietary recommendations to the long-term care patient on room {patient.get('room_number')} bed {patient.get('bed_number')} based on food intake record for the day {curdate}.

# Task:
# 1. Evaluate the patient's actual intake of {format_calculated_intakes(calculated_intake.get("aggregated_total", {}))} against the retrieved HPA Guidelines for all macros (Calories, Protein, Carbohydrates, Fats, Fiber).
# 2. Note any deficiencies or excesses based on the math remarks: {nutrition_remarks}.
# 3. Based on the evaluation, recommend 5 suitable meals from the provided list to help balance the patient's diet.

# Context:
# Patient Information:
# {get_patient_info(patient)}

# Food intakes for {curdate}:
# {food_intake_context}

# Available Meals:
# {meals_text}

# HPA Guidelines (Translate and apply these rules):
# Protein: {build_rag_context(query_1_results)}
# Carbohydrates: {build_rag_context(query_2_results)}
# Lipids/Fats: {build_rag_context(query_3_results)}
# Calories: {build_rag_context(query_4_results)}
# Fiber: {build_rag_context(query_5_results)}

# Rules:
# - If there is no food intake record for {curdate}, refrain from answering queries and politely explain that dietary recommendations cannot be provided because no intake was recorded for that day.
# - ONLY recommend meals that explicitly exist in the 'Available Meals' list.
# - Keep the response concise, professional, and under 200 words.
# - ONLY answer in English. Do not use Chinese characters in the final output.
# """
#             meal_recommendations = ask_llm(prompt)

#             # 4. SEND PROFESSIONAL JSON RESPONSE
#             response = {
                
#                 "response": meal_recommendations,
#                 "date": curdate,
#                 "patient": patient,
#                 "patient_dris": patient_dris,
#                 "food_intake_docs": food_intake_docs,
#                 "lunch_items": lunch_items,
#                 "lunch_nutritional_content": lunch_nutri_content,
#                 "dinner_items": dinner_items,
#                 "dinner_nutritional_content": dinner_nutri_content,
#                 "total_nutritional_content": total_nutri_content,
#                 "daily_nutrition_remarks": nutrition_remarks,
#             }
            
#             return Response(
#                 response,
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response({"detail": "Error generating response", "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # OKAY - LIVE DATA FOR THE ANALYSIS PAGE (Weekly Dropdown)
# class WeeklyRecommendationsByPatientView():
#     def get(self, request, pk):
#         try:
#             # 1. INFORMATION RETRIEVAL
#             # Patient Profile
#             patient_profile= get_patient_profile(pk)
#             patient = patient_profile[0][1]

#             # Dietary Targets
#             dietary_targets = get_patient_dietary_targets(pk)
#             patient_dris = dietary_targets[0][1]
#             patient_dris = {
#                 "calories_kcal": patient_dris["total_calories_kcal"],
#                 "protein_g": patient_dris["total_protein_g"],
#                 "fats_g": patient_dris["total_fats_g"],
#                 "carbohydrates_g": patient_dris["total_carbohydrates_g"],
#                 "fiber_g": patient_dris["total_fiber_g"],
#             }


#             # Food Intakes (Last 7 days)
#             # a. Current date
#             curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()

#             # b. Create a list of dates of the previous 7 days            
#             dates_list = [
#                 (curdate - timedelta(days=i)).strftime("%Y-%m-%d")
#                 for i in range(7)
#             ]
            
#             # c. Get intake docs (PAST 7-DAYS RECORDS)
#             weekly_intake_docs = [day_7_docs, day_6_docs, day_5_docs, day_4_docs, day_3_docs, day_2_docs, day_1_docs]

#             # d. Create list of docs with day number & dates
#             total_days = 7

#             weekly_data = {
#                 total_days - i: {
#                     "date": date,
#                     "food_intake_docs": docs
#                 }
#                 for i, (date, docs) in enumerate(zip(dates_list, weekly_intake_docs))
#             }


#             # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
#             for day_data in weekly_data.values():
#                 food_intake_docs = day_data["food_intake_docs"]

#                 # a. In each meal time (Lunch/Dinner), get intake volume per food class
#                 calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
                
#                     # Format Lunch for LLM (e.g, 40 ml of chicken, 30 ml of broccoli, etc.)
#                 lunch_intakes = calculated_intake['by_meal'].get('lunch')
#                 formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
#                 lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
#                     # Format Dinner for LLM
#                 dinner_intakes = calculated_intake['by_meal'].get('dinner')
#                 formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
#                 dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

                
#                 # b. In each meal time (Lunch/Dinner), get nutritional content of food intakes
                
#                     # Lunch Nutritional Content
#                 lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            
#                     # Dinner Nutritional Content
#                 dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            
#                 # c. Add Total Nutritional Content for both lunch & dinner 
#                 total_nutri_content = {
#                     "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
#                     "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
#                     "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
#                     "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
#                     "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
#                 } 

#                 # d. Get nutritional remarks
#                 nutrition_remarks = {
#                     "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "protein_g"),
#                     "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fats_g"),
#                     "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "carbohydrates_g"),
#                     "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fiber_g"),
#                     "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "calories_kcal"),
#                 }

#                 # e. Append results to each day in weekly_data 
#                 day_data["lunch_items"] = lunch_items
#                 day_data["lunch_nutritional_content"] = lunch_nutri_content
#                 day_data["dinner_items"] = dinner_items
#                 day_data["dinner_nutritional_content"] = dinner_nutri_content
#                 day_data["total_nutritional_content"] = total_nutri_content
#                 day_data["daily_nutrition_remarks"] = nutrition_remarks

            
#             # 3. WEEKLY AVERAGE
#             # a. Get valid days with food intake records (to only get average of days with records)
#             valid_days = [day_data for day_data in weekly_data.values() if day_data["food_intake_docs"]]

#             # b. Calculate weekly average for each nutrient based on valid days
#             if valid_days:
#                 weekly_average_nutri_content = {
#                     "calories_kcal": sum(day["total_nutritional_content"]["calories_kcal"] for day in valid_days) / len(valid_days),
#                     "protein_g": sum(day["total_nutritional_content"]["protein_g"] for day in valid_days) / len(valid_days),
#                     "fats_g": sum(day["total_nutritional_content"]["fats_g"] for day in valid_days) / len(valid_days),
#                     "carbohydrates_g": sum(day["total_nutritional_content"]["carbohydrates_g"] for day in valid_days) / len(valid_days),
#                     "fiber_g": sum(day["total_nutritional_content"]["fiber_g"] for day in valid_days) / len(valid_days),
#                 }
#             else:
#                 # c. If no valid days (i.e., no food intake records), set weekly average to 0 or None
#                 weekly_average_nutri_content = {
#                     "calories_kcal": 0,
#                     "protein_g": 0,
#                     "fats_g": 0,
#                     "carbohydrates_g": 0,
#                     "fiber_g": 0,
#                 }

#                 # d. GET NUTRITIONAL REMARKS BASED ON RECOMMENDED INTAKE AND WEEKLY AVERAGE INTAKE 
#             weekly_nutrition_remarks = {
#                 "protein_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "protein_g"),
#                 "fats_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fats_g"),
#                 "carbohydrates_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "carbohydrates_g"),
#                 "fiber_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fiber_g"),
#                 "calories_kcal": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "calories_kcal"),
#             }


#             # 4. GIVE WEEKLY MEAL RECOMMENDATIONS; CREATE PROMPT & GENERATE RESPONSE WITH LLM
#                 # a. Get meals from database
#             meal_names = meal_names_list 
#                 # b. Get meal recommendations
#             meal_recos = get_weekly_meal_recommendations(meal_names, weekly_nutrition_remarks)

            
#             # 5. FORMAT weekly_data FOR RESPONSE
#             for day_data in weekly_data.values():
#                 day_data.pop("food_intake_docs", None)


#             # 6. SEND RESPONSE
#             response = {
#                     "response": meal_recos,
#                     "date": curdate,
#                     "patient_dris": patient_dris,
#                     "weekly_data": weekly_data,
#                     "weekly_average_nutritional_content": weekly_average_nutri_content,
#                     "weekly_nutrition_remarks": weekly_nutrition_remarks,
#                 }
            
#             return Response(
#                 response,
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )

# OBSOLETE - Not used currently on the flow on the analysis page
# class DailyRecommendationsByDummyPatientView(APIView):    
#     def get(self, request):
#         try:
#             # 1. INFORMATION RETRIEVAL
#             # Current Day
#             # curdate = datetime.now().strftime("%Y-%m-%d")
#             curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
            
#             # Patient Profile
#             patient_profile_doc= dummy_patient_profile_doc
            
#             # Dietary Targets
#             patient_dris = dummy_recommended_intakes
            
#             # Food Intakes (Current date)
#             food_intake_docs = dummy_food_intake_docs


#             # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
#             # a. In each meal time (Lunch/Dinner), get food intake volume per food class
#             calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
            
#                 # Format Lunch for LLM (e.g, 40 ml of chicken, 30 ml of broccoli, etc.)
#             lunch_intakes = calculated_intake['by_meal'].get('lunch')
#             formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
#             lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            

#                 # Format Dinner for LLM
#             dinner_intakes = calculated_intake['by_meal'].get('dinner')
#             formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
#             dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

            
#             # b. In each meal time (Lunch/Dinner), get nutritional content of food intakes
            
#                 # Lunch Nutritional Content
#             lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)

#                 # Dinner Nutritional Content
#             dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)

#             # c. Add Total Nutritional Content for both lunch & dinner 
#             total_nutri_content = {
#                 "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
#                 "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
#                 "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
#                 "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
#                 "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
#             } 

#             # d. Get nutritional remarks
#             nutrition_remarks = {
#                 "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "protein_g"),
#                 "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fats_g"),
#                 "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "carbohydrates_g"),
#                 "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fiber_g"),
#                 "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "calories_kcal"),
#             }        


#             # 3. CREATE PROMPT & GENERATE RESPONSE WITH LLM; To give meal recommendations based on remarks
#             # a. Get meals from database
#             meal_names = meal_names_list

#             # b. Get meal recommendations
#             meal_recommendations = get_daily_meal_recommendations(meal_names, nutrition_remarks)


#             # 4. SEND RESPONSE
#             response = {
#                     "response": meal_recommendations,
#                     "date": curdate,
#                     "patient_dris": patient_dris,
#                     "lunch_items": lunch_items,
#                     "lunch_nutritional_content": lunch_nutri_content,
#                     "dinner_items": dinner_items,
#                     "dinner_nutritional_content": dinner_nutri_content,
#                     "total_nutritional_content": total_nutri_content,
#                     "daily_nutrition_remarks": nutrition_remarks,
#                 }

#             return Response(
#                 response,
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
        
# OBSOLETE - Not used currently on the flow on the analysis page
# class WeeklyRecommendationsByDummyPatientView(APIView):    
#     def get(self, request):
#         try:
#             # 1. INFORMATION RETRIEVAL
#             # Patient Profile
#             patient_profile_doc= dummy_patient_profile_doc
            
#             # Dietary Targets
#             patient_dris = dummy_recommended_intakes
            
#             # Food Intakes (Last 7 days)
#             # a. Current date
#             curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()

#             # b. Create a list of dates of the previous 7 days            
#             dates_list = [
#                 (curdate - timedelta(days=i)).strftime("%Y-%m-%d")
#                 for i in range(7)
#             ]
            
#             # c. Get intake docs (PAST 7-DAYS RECORDS)
#             weekly_intake_docs = [day_7_docs, day_6_docs, day_5_docs, day_4_docs, day_3_docs, day_2_docs, day_1_docs]

#             # d. Create list of docs with day number & dates
#             total_days = 7

#             weekly_data = {
#                 total_days - i: {
#                     "date": date,
#                     "food_intake_docs": docs
#                 }
#                 for i, (date, docs) in enumerate(zip(dates_list, weekly_intake_docs))
#             }


#             # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
#             for day_data in weekly_data.values():
#                 food_intake_docs = day_data["food_intake_docs"]

#                 # a. In each meal time (Lunch/Dinner), get intake volume per food class
#                 calculated_intake = calculate_food_item_intake(food_intake_docs, debug=True)
                
#                     # Format Lunch for LLM (e.g, 40 ml of chicken, 30 ml of broccoli, etc.)
#                 lunch_intakes = calculated_intake['by_meal'].get('lunch')
#                 formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
#                 lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
#                     # Format Dinner for LLM
#                 dinner_intakes = calculated_intake['by_meal'].get('dinner')
#                 formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
#                 dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

                
#                 # b. In each meal time (Lunch/Dinner), get nutritional content of food intakes
                
#                     # Lunch Nutritional Content
#                 lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            
#                     # Dinner Nutritional Content
#                 dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            
#                 # c. Add Total Nutritional Content for both lunch & dinner 
#                 total_nutri_content = {
#                     "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
#                     "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
#                     "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
#                     "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
#                     "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
#                 } 

#                 # d. Get nutritional remarks
#                 nutrition_remarks = {
#                     "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "protein_g"),
#                     "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fats_g"),
#                     "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "carbohydrates_g"),
#                     "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fiber_g"),
#                     "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "calories_kcal"),
#                 }

#                 # e. Append results to each day in weekly_data 
#                 day_data["lunch_items"] = lunch_items
#                 day_data["lunch_nutritional_content"] = lunch_nutri_content
#                 day_data["dinner_items"] = dinner_items
#                 day_data["dinner_nutritional_content"] = dinner_nutri_content
#                 day_data["total_nutritional_content"] = total_nutri_content
#                 day_data["daily_nutrition_remarks"] = nutrition_remarks

            
#             # 3. WEEKLY AVERAGE
#             # a. Get valid days with food intake records (to only get average of days with records)
#             valid_days = [day_data for day_data in weekly_data.values() if day_data["food_intake_docs"]]

#             # b. Calculate weekly average for each nutrient based on valid days
#             if valid_days:
#                 weekly_average_nutri_content = {
#                     "calories_kcal": sum(day["total_nutritional_content"]["calories_kcal"] for day in valid_days) / len(valid_days),
#                     "protein_g": sum(day["total_nutritional_content"]["protein_g"] for day in valid_days) / len(valid_days),
#                     "fats_g": sum(day["total_nutritional_content"]["fats_g"] for day in valid_days) / len(valid_days),
#                     "carbohydrates_g": sum(day["total_nutritional_content"]["carbohydrates_g"] for day in valid_days) / len(valid_days),
#                     "fiber_g": sum(day["total_nutritional_content"]["fiber_g"] for day in valid_days) / len(valid_days),
#                 }
#             else:
#                 # c. If no valid days (i.e., no food intake records), set weekly average to 0 or None
#                 weekly_average_nutri_content = {
#                     "calories_kcal": 0,
#                     "protein_g": 0,
#                     "fats_g": 0,
#                     "carbohydrates_g": 0,
#                     "fiber_g": 0,
#                 }

#                 # d. GET NUTRITIONAL REMARKS BASED ON RECOMMENDED INTAKE AND WEEKLY AVERAGE INTAKE 
#             weekly_nutrition_remarks = {
#                 "protein_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "protein_g"),
#                 "fats_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fats_g"),
#                 "carbohydrates_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "carbohydrates_g"),
#                 "fiber_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fiber_g"),
#                 "calories_kcal": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "calories_kcal"),
#             }


#             # 4. GIVE WEEKLY MEAL RECOMMENDATIONS; CREATE PROMPT & GENERATE RESPONSE WITH LLM
#                 # a. Get meals from database
#             meal_names = meal_names_list
#                 # b. Get meal recommendations
#             meal_recos = get_weekly_meal_recommendations(meal_names, weekly_nutrition_remarks)

            
#             # 5. FORMAT weekly_data FOR RESPONSE
#             for day_data in weekly_data.values():
#                 day_data.pop("food_intake_docs", None)


#             # 6. SEND RESPONSE
#             response = {
#                     "response": meal_recos,
#                     "date": curdate,
#                     "patient_dris": patient_dris,
#                     "weekly_data": weekly_data,
#                     "weekly_average_nutritional_content": weekly_average_nutri_content,
#                     "weekly_nutrition_remarks": weekly_nutrition_remarks,
#                 }

#             return Response(
#                 response,
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
        
# OBSOLETE - Not used currently on the flow on the analysis page; LIVE MONTHLY DATA IS NOT USED CURRENTLY, ONLY LAST 7 DAYS DATA IS USED FOR WEEKLY RECOMMENDATION AND ANALYSIS
# class MonthlyRecommendationsByDummyPatientView(APIView):    
#     def get(self, request):
#         try:
#             # 1. INFORMATION RETRIEVAL
#             # Patient Profile
#             patient_profile_doc= dummy_patient_profile_doc
            
#             # Dietary Targets
#             patient_dris = dummy_recommended_intakes
            
#             # Food Intakes (Last 28 days)
#             total_days = 28
#             # a. Current date
#             curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()

#             # b. Create a list of dates of the previous 28 days            
#             dates_list = [
#                 (curdate - timedelta(days=i)).strftime("%Y-%m-%d")
#                 for i in range(total_days)
#             ]
            
#             # c. Get intake docs (PAST 28-DAYS RECORDS)
#             monthly_intake_docs = ([day_7_docs, day_6_docs, day_5_docs, day_4_docs, day_3_docs, day_2_docs, day_1_docs] * 4)

#             # d. Create list of docs with day number & dates
#             monthly_data = {
#                 total_days - i: {
#                     "date": date,
#                     "food_intake_docs": docs
#                 }
#                 for i, (date, docs) in enumerate(zip(dates_list, monthly_intake_docs))
#             }


#             # 2. FOOD VOLUME -> NUTRITIONAL CONTENT
#             for day_data in monthly_data.values():
#                 food_intake_docs = day_data["food_intake_docs"]

#                 # a. In each meal time (Lunch/Dinner), get intake volume per food class
#                 calculated_intake = calculate_food_item_intake(food_intake_docs, debug=False)
                
#                     # Format Lunch for LLM (e.g, 40 ml of chicken, 30 ml of broccoli, etc.)
#                 lunch_intakes = calculated_intake['by_meal'].get('lunch')
#                 formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
#                 lunch_items = format_calculated_intakes_for_response(lunch_intakes) if lunch_intakes else None
            
#                     # Format Dinner for LLM
#                 dinner_intakes = calculated_intake['by_meal'].get('dinner')
#                 formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None
#                 dinner_items = format_calculated_intakes_for_response(dinner_intakes) if dinner_intakes else None

                
#                 # b. In each meal time (Lunch/Dinner), get nutritional content of food intakes
                
#                     # Lunch Nutritional Content
#                 lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            
#                     # Dinner Nutritional Content
#                 dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            
#                 # c. Add Total Nutritional Content for both lunch & dinner 
#                 total_nutri_content = {
#                     "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
#                     "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
#                     "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
#                     "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
#                     "fiber_g": lunch_nutri_content["fiber_g"] + dinner_nutri_content["fiber_g"],
#                 } 

#                 # d. Get nutritional remarks
#                 nutrition_remarks = {
#                     "protein_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "protein_g"),
#                     "fats_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fats_g"),
#                     "carbohydrates_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "carbohydrates_g"),
#                     "fiber_g": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "fiber_g"),
#                     "calories_kcal": get_nutrition_remarks(patient_dris, total_nutri_content, nutrient = "calories_kcal"),
#                 }

#                 # e. Append results to each day in weekly_data 
#                 day_data["lunch_items"] = lunch_items
#                 day_data["lunch_nutritional_content"] = lunch_nutri_content
#                 day_data["dinner_items"] = dinner_items
#                 day_data["dinner_nutritional_content"] = dinner_nutri_content
#                 day_data["total_nutritional_content"] = total_nutri_content
#                 day_data["daily_nutrition_remarks"] = nutrition_remarks

            
#             # 3. WEEKLY AVERAGE (4 weeks)
#             # a. Split 28 days into 4 weeks
#             # Week 1 → days 1–7 (oldest)
#             # Week 4 → days 22–28 (latest)
#             monthly_list = list(monthly_data.values())
#             # Reverse so oldest → latest
#             monthly_list.reverse()
#             weeks = [
#                 monthly_list[i:i+7]
#                 for i in range(0, len(monthly_list), 7)
#             ]
            
#             weekly_data = {}

#             for i, week in enumerate(weeks, start=1):
#                 # b. Get valid days with food intake records (to only get average of days with records)
#                 valid_days = [day for day in week if day.get("food_intake_docs")]
            
#                 # c. Calculate weekly average for each nutrient based on valid days
#                 num = len(valid_days)
#                 if valid_days:
#                     weekly_average_nutri_content = {
#                         "calories_kcal": sum(day["total_nutritional_content"]["calories_kcal"] for day in valid_days) / num,
#                         "protein_g": sum(day["total_nutritional_content"]["protein_g"] for day in valid_days) / num,
#                         "fats_g": sum(day["total_nutritional_content"]["fats_g"] for day in valid_days) / num,
#                         "carbohydrates_g": sum(day["total_nutritional_content"]["carbohydrates_g"] for day in valid_days) / num,
#                         "fiber_g": sum(day["total_nutritional_content"]["fiber_g"] for day in valid_days) / num,
#                     }
#                 else:
#                     # d. If no valid days (i.e., no food intake records), set weekly average to 0 or None
#                     weekly_average_nutri_content = {
#                         "calories_kcal": 0,
#                         "protein_g": 0,
#                         "fats_g": 0,
#                         "carbohydrates_g": 0,
#                         "fiber_g": 0,
#                     }
            
#                     # d. GET NUTRITIONAL REMARKS BASED ON RECOMMENDED INTAKE AND WEEKLY AVERAGE INTAKE 
#                 weekly_nutrition_remarks = {
#                     "protein_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "protein_g"),
#                     "fats_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fats_g"),
#                     "carbohydrates_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "carbohydrates_g"),
#                     "fiber_g": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "fiber_g"),
#                     "calories_kcal": get_nutrition_remarks(patient_dris, weekly_average_nutri_content, nutrient = "calories_kcal"),
#                 }

#                 weekly_data[i] = {
#                     "weekly_average_nutritional_content": weekly_average_nutri_content,
#                     "weekly_nutrition_remarks": weekly_nutrition_remarks
#                 }

#             # 4. MONTHLY AVERAGE
#             # a. Get valid weeks with food intake records (to only get average of weeks with records)
#             valid_weeks = [    
#                 week for week in weekly_data.values()
#                 if week["weekly_average_nutritional_content"]["protein_g"] != 0
#             ]

#             # b. Calculate monthly average for each nutrient based on valid days
#             num = len(valid_weeks)
#             if valid_weeks:
#                 monthly_average_nutri_content = {
#                     "calories_kcal": sum(week["weekly_average_nutritional_content"]["calories_kcal"] for week in valid_weeks) / num,
#                     "protein_g": sum(week["weekly_average_nutritional_content"]["protein_g"] for week in valid_weeks) / num,
#                     "fats_g": sum(week["weekly_average_nutritional_content"]["fats_g"] for week in valid_weeks) / num,
#                     "carbohydrates_g": sum(week["weekly_average_nutritional_content"]["carbohydrates_g"] for week in valid_weeks) / num,
#                     "fiber_g": sum(week["weekly_average_nutritional_content"]["fiber_g"] for week in valid_weeks) / num,
#                 }
#             else:
#                 # c. If no valid weeks (i.e., no intakes & nutrients), set monthly average to 0 or None
#                 monthly_average_nutri_content = {
#                     "calories_kcal": 0,
#                     "protein_g": 0,
#                     "fats_g": 0,
#                     "carbohydrates_g": 0,
#                     "fiber_g": 0,
#                 }

#                 # d. GET NUTRITIONAL REMARKS BASED ON RECOMMENDED INTAKE AND MONTHLY AVERAGE INTAKE 
#             monthly_nutrition_remarks = {
#                 "protein_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "protein_g"),
#                 "fats_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "fats_g"),
#                 "carbohydrates_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "carbohydrates_g"),
#                 "fiber_g": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "fiber_g"),
#                 "calories_kcal": get_nutrition_remarks(patient_dris, monthly_average_nutri_content, nutrient = "calories_kcal"),
#             }


#             # 4. GIVE MONTHLY MEAL RECOMMENDATIONS; CREATE PROMPT & GENERATE RESPONSE WITH LLM
#                 # a. Get meals from database
#             meal_names = meal_names_list
#                 # b. Get meal recommendations
#             meal_recos = get_monthly_meal_recommendations(meal_names, monthly_nutrition_remarks)

            
#             # 5. FORMAT monthly_data FOR RESPONSE
#             for day_data in monthly_data.values():
#                 day_data.pop("food_intake_docs", None)


#             # 6. SEND RESPONSE
#             response = {
#                 "response": meal_recos,
#                 "date": curdate,
#                 "patient_dris": patient_dris,
#                 "monthly_data": monthly_data,
#                 "weekly_data": weekly_data,
#                 "monthly_average_nutritional_content": monthly_average_nutri_content,
#                 "monthly_nutrition_remarks": monthly_nutrition_remarks,
#             }

#             return Response(
#                 response,
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )


# !!!!OBSOLETE - Not used currently on the flow on the analysis page
# Daily Intake recommender
# class DailyPatientFoodIntakeRecommenderView(APIView):    
#     def get(self, request, pk):
#         try:
#             curdate = datetime.now().strftime("%Y-%m-%d")
#             # curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

#             # 1. INFORMATION RETRIEVAL
#             # 1.1 Get patient info & patient profile doc
#             patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
#             patient_id = patient.get("id")
#             room_number = patient.get("room_number")
#             bed_number = patient.get("bed_number") 
#             age = patient.get("age")
#             sex = patient.get("sex")
#             # patient_profile_doc = get_patient_profile(pk)[0]
#             patient_profile_doc= get_patient_info(patient)

#             # 1.2 Get patient dietary target doc
#             dri_target_doc = get_patient_dietary_targets(patient_id)[0]

#             # 1.3 Get Food Intake docs
#                 # a. Before and After Segmented Meal Records
#             food_intake_res = get_patient_segmented_intake(patient_id, curdate)

#                 # b. Build food intake context for prompt
#             food_intake_context = "\n\n - ".join([doc for doc, metadata in food_intake_res])

#                 # c. Format food intake docs for response
#             food_intake_docs = format_food_intakes_docs(food_intake_res)

#                 # d. Calculate food intake per item in volume (mL) based on before and after meal data.
#             food_item_intakes = calculate_food_item_intake(food_intake_docs)
#             aggregated_total_intakes = food_item_intakes.get("aggregated_total", {})
#             formatted_aggregated_total_intakes = format_calculated_intakes(aggregated_total_intakes)


#             # 1.4 HPA docs retrieval using queries (MUST BE CHINESE FOR QDRANT TO MATCH)
#             if not age and not sex:
#                 descriptor = "高齡長者" 
#             elif age and sex:
#                 descriptor = f"{age}歲的{sex}" 
#             elif age:
#                 descriptor = f"{age}歲"
#             else:
#                 descriptor = sex

#             # Fetch ALL 5 Macros
#             query_1_results = retrieve_all(f"{descriptor}的每日蛋白質攝取建議與需求", top_k=3)
#             query_2_results = retrieve_all(f"{descriptor}的每日碳水化合物攝取建議與需求", top_k=3)
#             query_3_results = retrieve_all(f"{descriptor}的每日脂質攝取建議與需求", top_k=3)
#             query_4_results = retrieve_all(f"{descriptor}的每日熱量(Calories)攝取建議與需求", top_k=3)
#             query_5_results = retrieve_all(f"{descriptor}的每日膳食纖維(Fiber)攝取建議與需求", top_k=3)

#             # 2. CREATE PROMPT (LLM Instructions in English)
#             prompt = f"""
# Based on the following information, evaluate the patient's diet and provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number} based on food intake record for the day {curdate}.

# Task:
# 1. Evaluate the patient's actual intake of {formatted_aggregated_total_intakes} against the retrieved HPA Guidelines for all macros (Calories, Protein, Carbohydrates, Fats, Fiber).
# 2. Note any deficiencies or excesses based on the math.

# Context:
# Patient Information:
# {patient_profile_doc}

# Food intakes for {curdate}:
# {food_intake_context}

# HPA Guidelines (Translate and apply these rules):
# Protein: {build_rag_context(query_1_results)}
# Carbohydrates: {build_rag_context(query_2_results)}
# Lipids/Fats: {build_rag_context(query_3_results)}
# Calories: {build_rag_context(query_4_results)}
# Fiber: {build_rag_context(query_5_results)}

# Rules:
# - If there is no food intake record for {curdate}, refrain from answering queries and politely explain that dietary recommendations cannot be provided because no intake was recorded for that day.
# - Keep the response concise, professional, and under 200 words.
# - ONLY answer in English. Do not use Chinese characters in the final output.
# """

#             # 3. GENERATE RESPONSE WITH LLM
#             response = ask_llm(prompt)

#             return Response(
#                 {
#                     "response": response,
#                     "final_prompt": prompt,
#                     "patient_profile_doc": patient_profile_doc,
#                     "dri_target_doc": dri_target_doc,
#                     "food_intake_docs": food_intake_docs,
#                     "food_item_intakes": food_item_intakes,
#                     "query_1_results": query_1_results,
#                     "query_2_results": query_2_results,
#                     "query_3_results": query_3_results,
#                     "date": curdate,
#                     "room_number": room_number,
#                     "bed_number": bed_number,
#                 },
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
        
# OBSOLETE - Not used currently on the flow on the analysis page; LIVE WEEKLY DATA IS NOT USED CURRENTLY, ONLY LAST 7 DAYS DATA IS USED FOR WEEKLY RECOMMENDATION AND ANALYSIS
# Weekly Intake recommender with the ingredients & nutrients
class WeeklyPatientFoodIntakeRecommenderView(APIView):    
    def get(self, request, pk):
        try:
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 

            # Current date
            curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()

            # Previous 6 days            
            curdate_str      = curdate.strftime("%Y-%m-%d")
            day_minus_1_str  = (curdate - timedelta(days=1)).strftime("%Y-%m-%d")
            day_minus_2_str  = (curdate - timedelta(days=2)).strftime("%Y-%m-%d")
            day_minus_3_str  = (curdate - timedelta(days=3)).strftime("%Y-%m-%d")
            day_minus_4_str  = (curdate - timedelta(days=4)).strftime("%Y-%m-%d")
            day_minus_5_str  = (curdate - timedelta(days=5)).strftime("%Y-%m-%d")
            day_minus_6_str  = (curdate - timedelta(days=6)).strftime("%Y-%m-%d")


            # Get relevant patient docs
            #food_intakes
            curdate_results = get_patient_food_intake(patient_id, day_minus_5_str)
            day_minus_1_results = get_patient_food_intake(patient_id, day_minus_1_str)
            day_minus_2_results = get_patient_food_intake(patient_id, day_minus_2_str)
            day_minus_3_results = get_patient_food_intake(patient_id, day_minus_3_str)
            day_minus_4_results = get_patient_food_intake(patient_id, day_minus_4_str)
            day_minus_5_results = get_patient_food_intake(patient_id, day_minus_5_str)
            day_minus_6_results = get_patient_food_intake(patient_id, day_minus_6_str)
            
            #dri_targets
            dri_results = get_patient_dietary_targets(patient_id,)

            #patient_profile
            patient_info = get_patient_info(patient)

            # Hpa context
            hpa_macros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的巨量營養素攝取建議，包括脂質、蛋白質與碳水化合物的營養需求。
"""
            hpa_macros_results = retrieve_all(hpa_macros_query, top_k=7)

            hpa_micros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的微量營養素攝取建議，包括維生素與礦物質的營養需求。
"""
            hpa_micros_results = retrieve_all(hpa_micros_query, top_k=7)

            
            # Chunks
            weekly_food_intakes_debug = {
                curdate_str: format_food_intakes_chunks(curdate_results),
                day_minus_1_str: format_food_intakes_chunks(day_minus_1_results),
                day_minus_2_str: format_food_intakes_chunks(day_minus_2_results),
                day_minus_3_str: format_food_intakes_chunks(day_minus_3_results),
                day_minus_4_str: format_food_intakes_chunks(day_minus_4_results),
                day_minus_5_str: format_food_intakes_chunks(day_minus_5_results),
                day_minus_6_str: format_food_intakes_chunks(day_minus_6_results),
            }

            hpa_macros_chunks = hpa_macros_results
            hpa_micros_chunks = hpa_micros_results

            # Build context 
            curdate_context = "\n\n- ".join([doc for doc, metadata in curdate_results]) if curdate_results else "Missing"
            day_minus_1_context = "\n\n- ".join([doc for doc, metadata in day_minus_1_results]) if day_minus_1_results else "Missing"
            day_minus_2_context = "\n\n- ".join([doc for doc, metadata in day_minus_2_results]) if day_minus_2_results else "Missing"
            day_minus_3_context = "\n\n- ".join([doc for doc, metadata in day_minus_3_results]) if day_minus_3_results else "Missing"
            day_minus_4_context = "\n\n- ".join([doc for doc, metadata in day_minus_4_results]) if day_minus_4_results else "Missing"
            day_minus_5_context = "\n\n- ".join([doc for doc, metadata in day_minus_5_results]) if day_minus_5_results else "Missing"
            day_minus_6_context = "\n\n- ".join([doc for doc, metadata in day_minus_6_results]) if day_minus_6_results else "Missing"

            hpa_macros_context = build_rag_context(hpa_macros_results)
            hpa_micros_context = build_rag_context(hpa_micros_results)

            # Build prompt
            prompt = f"""
Based on the following information and food intake records in the last week, please provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number}.

相關資訊：
Patient Information:
{patient_info}

Patient dietary targets:
{dri_results}

Patient food intake context:
6 Days ago ({day_minus_6_str}): 
{day_minus_6_context}

5 Days ago ({day_minus_5_str}): 
{day_minus_5_context}

4 Days ago ({day_minus_4_str}): 
{day_minus_4_context}

3 Days ago ({day_minus_3_str}): 
{day_minus_3_context}

2 Days ago ({day_minus_2_str}): 
{day_minus_2_context}

Yesterday ({day_minus_1_str}): 
{day_minus_1_context}

Today ({curdate}):
{curdate_context}

Nutrition guidelines context:
Macronutrients:
{hpa_macros_context}

Micronutrients:
{hpa_micros_context}

回應規則：
- Identify any patterns in the patient's food intake.
- If only some days are missing, provide recommendations based on the available records or general recommendationsand mention the missing days.
- If all the food intake record is missing for today, yesteday, and 2, 3, 4, 5, and 6 days ago, politely explain that dietary recommendations for the week cannot be provided.
- 請言簡意賅。回覆字數應少於400個字。
- 若資訊中無明確答案，請回應「無可用資訊
- 請僅以繁體中文回覆。
"""

            # Pass context
            # response = ask_llm_with_token_limit(prompt, token_limit=200)
            response = ask_llm(prompt)
            # response = ask_ollama_llm(prompt)
            return Response(
                {
                    "response": response,
                    "weekly_food_intakes": weekly_food_intakes_debug,
                    "dri_results": dri_results,
                    "hpa_macros_chunks": hpa_macros_chunks,
                    "hpa_micros_chunks": hpa_micros_chunks,
                    "date": curdate,
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
        

# OBSOLETE - Not used currently on the flow on the analysis page; LIVE WEEKLY DATA IS NOT USED CURRENTLY, ONLY LAST 7 DAYS DATA IS USED FOR WEEKLY RECOMMENDATION AND ANALYSIS
# Monthly Intake recommender with the ingredients & nutrients
# class MonthlyPatientFoodIntakeRecommenderView(APIView):    
#     def get(self, request, pk):
#         try:
#             patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
#             patient_id = patient.get("id")
#             room_number = patient.get("room_number")
#             bed_number = patient.get("bed_number") 

#             # Current date, month, year
#             curmonth = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%m")
#             cur_month_name = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%B")

#             curyear = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y")

#             curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()
#             no_of_days_in_curmonth = calendar.monthrange(curdate.year, curdate.month)[1]
#             list_of_days = list(range(1, no_of_days_in_curmonth + 1))
#             dates_in_curmonth_list = get_dates_in_current_month()


#             # Get patient docs on cur month
#             fi_results_in_curmonth_list = get_food_intake_results_in_curmonth(dates_in_curmonth_list, patient_id)
#             dri_results = get_patient_dietary_targets(patient_id,)
#             patient_info = get_patient_info(patient)

#             # Hpa context
#             hpa_macros_query = f"""
# 高齡{patient.get("age")}歲的{patient.get("sex")}長者的巨量營養素攝取建議，包括脂質、蛋白質與碳水化合物的營養需求。
# """
#             hpa_macros_results = retrieve_all(hpa_macros_query, top_k=7)

#             hpa_micros_query = f"""
# 高齡{patient.get("age")}歲的{patient.get("sex")}長者的微量營養素攝取建議，包括維生素與礦物質的營養需求。
# """
#             hpa_micros_results = retrieve_all(hpa_micros_query, top_k=7)

#             hpa_gen_query = """
# 長者的營養建議、指引或飲食建議。
# """
#             hpa_gen_results = retrieve_text(hpa_gen_query, top_k=7)

#             # chunks
#             hpa_macros_chunks = hpa_macros_results
#             hpa_micros_chunks = hpa_micros_results

#             # Build context 
#             curmonth_food_intake_context = create_monthly_food_intake_context(fi_results_in_curmonth_list)
#             hpa_macros_context = build_rag_context(hpa_macros_results)
#             hpa_micros_context = build_rag_context(hpa_micros_results)
#             hpa_gen_context = build_rag_context(hpa_gen_results)

#             # Build prompt
#             prompt = f"""
# Based on the following information and food intake records in this month of {cur_month_name} {curyear}, please provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number}.

# 相關資訊：
# Patient Information:
# {patient_info}

# Patient daily dietary targets:
# {dri_results}

# {cur_month_name} food intake context:
# {curmonth_food_intake_context}

# Nutrition guidelines context:
# General:
# {hpa_gen_context}

# Macronutrients:
# {hpa_macros_context}

# Micronutrients:
# {hpa_micros_context}

# 回應規則：
# - Clearly state that these are dietary recommendations for {cur_month_name}.
# - Identify any patterns in the patient's food intake.
# - If some days are missing, provide recommendations based on the available records and briefly mention the missing days or weeks in the month.
# - If ALL the food intake records is missing for the {no_of_days_in_curmonth} in {cur_month_name} {curyear}, politely explain that dietary recommendations for the month cannot be provided.
# - Give safe general answers
# - 請言簡意賅。回覆字數應少於400個字。
# - 若資訊中無明確答案，請回應「無可用資訊
# - 請僅以繁體中文回覆。
# - 必要時請自行推論。
# """

#             # Pass context
#             # response = ask_llm_with_token_limit(prompt, token_limit=200)
#             response = ask_llm(prompt)
#             # response = ask_ollama_llm(prompt)
#             return Response(
#                 {
#                     "response": response,
#                     "monthly_food_intakes": fi_results_in_curmonth_list,
#                     "dri_results": dri_results,
#                     "hpa_macros_chunks": hpa_macros_chunks,
#                     "hpa_micros_chunks": hpa_micros_chunks,
#                     "date": curdate,
#                 },
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )


#!!! STANDBY BUT OKAY !!! - Not used currently on the flow on the analysis page
# Food Intake recommender BY DATE
# class PatientFoodIntakeRecommenderByDateView(APIView):    
#     def get(self, request, pk, date):
#         try:
#             # 1. INFORMATION RETRIEVAL
#             # 1.1 Get patient info & patient profile doc
#             patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
#             patient_id = patient.get("id")
#             room_number = patient.get("room_number")
#             bed_number = patient.get("bed_number") 
#             age = patient.get("age")
#             sex = patient.get("sex")
#             # patient_profile_doc = get_patient_profile(pk)[0]
#             patient_profile_doc= get_patient_info(patient)

#             # 1.2 Get patient dietary target doc
#             dri_target_doc = get_patient_dietary_targets(patient_id)[0]

#             # 1.3 Get Food Intake docs
#                 # a. Before and After Segmented Meal Records
#             food_intake_res = get_patient_segmented_intake(patient_id, date,)

#                 # b. Build food intake context for prompt
#             food_intake_context = "\n\n - ".join([doc for doc, metadata in food_intake_res])

#                 # c. Format food intake docs for response
#             food_intake_docs = format_food_intakes_docs(food_intake_res)

#                 # d. Calculate food intake per item in volume (mL) based on before and after meal data.
#             food_item_intakes = calculate_food_item_intake(food_intake_docs)
#             aggregated_total_intakes = food_item_intakes.get("aggregated_total", {})
#             formatted_aggregated_total_intakes = format_calculated_intakes(aggregated_total_intakes)


#             # 1.4 HPA docs retrieval using queries
#             if not age and not sex:
#                 descriptor = "elderly"
#             elif age and sex:
#                 descriptor = f"{age} year old {sex}"
#             elif age:
#                 descriptor = f"{age} year old"
#             else:
#                 descriptor = sex

#                 # a. Protein-related Query
#             query_1 = f"""Calculate the combined protein content of {formatted_aggregated_total_intakes} and determine if it meets the daily protein requirement for a {descriptor} patient?"""
#             query_1_results = retrieve_all(query_1, top_k= 5)
#             query_1_results_context = build_rag_context(query_1_results)

#                 # b. Carbohydrate-related Query
#             query_2 = f"""Calculate the combined carbohydrate content of {formatted_aggregated_total_intakes} and determine if it meets the daily carbohydrate requirement for a {descriptor} patient?"""
#             query_2_results = retrieve_all(query_2, top_k= 5)
#             query_2_results_context = build_rag_context(query_2_results)

#                 # c. Lipid-related Query
#             query_3 = f"""Calculate the combined lipid content of {formatted_aggregated_total_intakes} and determine if it meets the daily lipid requirement for a {descriptor} patient?"""
#             query_3_results = retrieve_all(query_3, top_k= 5)
#             query_3_results_context = build_rag_context(query_3_results)


#             # 2. CREATE PROMPT
#             prompt = f"""
# Based on the following information, answer the following queries to provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number} based on food intake record for the day {date}.

# Queries:
# - Query 1: {query_1}
# - Query 2: {query_2}
# - Query 3: {query_3}

# 相關資訊：
# Patient Information:
# {patient_profile_doc}

# Food intakes for {date}:
# {food_intake_context}

# Query 1 Context:
# {query_1_results_context}
# Query 2 Context:
# {query_2_results_context}
# Query 3 Context:
# {query_3_results_context}


# 回應規則：
# - If there are no food intake records for {date}, refrain from answering queries and politely explain that dietary recommendations cannot be provided because no intake was recorded for that day
# - Even if there is one meal entry, give recommendations.
# - Answer in English.
# """

#             # 3. GENERATE RESPONSE WITH LLM
#             response = ask_llm(prompt)

#             return Response(
#                 {
#                     "response": response,
#                     "final_prompt": prompt,
#                     "patient_profile_doc": patient_profile_doc,
#                     "dri_target_doc": dri_target_doc,
#                     "food_intake_docs": food_intake_docs,
#                     "food_item_intakes": food_item_intakes,
#                     "query_1_results": query_1_results,
#                     "query_2_results": query_2_results,
#                     "query_3_results": query_3_results,
#                     "date": date,
#                     "room_number": room_number,
#                     "bed_number": bed_number,
#                 },
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response(
#                 {
#                     "detail": "Error generating response", 
#                     "error": str(e)
#                 },
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )





# ============================
# DUMMY DATA
# ============================
dummy_food_intake_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]


dummy_patient_profile_doc = {
    "page_content": "The LTC Patient in Room 101, Bed 2 is a 75-year-old female. They have a BMI of 23.48 and an activity level classified as Low Active.",
    "metadata":
    {
        "doc_type": "patient_profile",
        "room_number": "101",
        "bed_number": "2",
        "ltc_patient_id": 2
    }
}


dummy_dietary_target_doc = {
    "page_content": "The LTC Patient in Room 101, Bed 2 requires a daily dietary target of: 1956.0 kcal, 40.7g of protein, 269.0g of carbohydrates, and 59.9g of fat. Their total water requirement is 2.71ml.",
    "metadata":
    {
        "doc_type": "dietary_target",
        "room_number": "101",
        "bed_number": "2",
        "ltc_patient_id": 2,
    }
}

dummy_recommended_intakes = {
    "calories_kcal": 1956.0, 
    "protein_g": 40.7, 
    "fats_g": 59.9, 
    "carbohydrates_g": 269.0,
    "fiber_g": 30.0
}


# From actual db
meal_names_list = ['滷肉排', '馬鈴薯炒肉末', '蒜醬麵腸', '炒時蔬', '筍絲豆皮湯', '滷油干魚', '滷豆支', '四季豆炒香腸', '炒時蔬', '紫菜蛋花湯', '什錦烏龍麵', '炒時蔬', '大白菜豆皮湯', '高麗菜粥', '馬鈴薯燉肉', '芋頭蛋黃球', '滷蘿蔔', '炒時蔬', '味噌蛋花湯', '滷雞排', '炒冬粉', '薑絲海帶根', '炒時蔬', '玉米湯', '洋蔥炒豬柳', '燴咖哩', '炒時蔬', '海帶芽湯', '銀斑魚', '沙茶素腰花', '玉米炒蛋', '炒時蔬', '高麗菜湯', '滷雞腿', '紅蘿蔔滷貢丸', '蒜醬百頁豆腐', '炒時蔬', '紫菜蛋花湯', '紅燒獅子頭', '蘿蔔滷豆輪', '馬鈴薯炒蛋', '炒時蔬', '玉米湯', '滷肉排', '海苔丸', '醬拌豆干', '炒時蔬', '冬菜豆芽湯', '碗粿', '筍絲豆皮湯', '皮蛋鹹粥', '日式豬排', '滷麵筋', '薑絲炒木耳', '炒時蔬', '海帶芽湯', '古早味炒麵', '蘿蔔湯', '絲瓜鹹粥', '滷雞排', '紅蘿蔔炒蛋', '馬鈴薯炒肉末', '炒時蔬', '玉米湯', '滷油干魚', '肉末滷油豆腐', '茄汁炒蛋', '炒時蔬', '筍絲豆皮湯', '香腸', '炒冬粉', '馬鈴薯炒肉末', '炒時蔬', '紫菜蛋花湯', '三杯里肌', '洋蔥炒甜不辣', '紅蘿蔔炒蛋', '炒時蔬', '冬菜豆芽湯', '滷雞腿', '蘿蔔滷豆輪', '炒時蔬', '味噌蛋花湯', '什錦米粉', '炒時蔬', '大白菜豆皮湯', '芋頭鹹粥', '滷銀斑魚', '滷筍絲', '蒜醬百頁豆腐', '炒時蔬', '海帶芽湯', '沙茶腿排', '三杯麵腸', '玉米炒蛋', '炒時蔬', '鳳梨苦瓜湯', '紅燒獅子頭', '滷海帶結', '炒時蔬', '玉米湯', '馬鈴薯燉肉', '滷蘿蔔', '炒冬粉', '炒時蔬', '筍絲豆皮湯', '炸無骨雞排', '燴咖哩', '紅蘿蔔炒蛋', '炒時蔬', '紫菜湯', '米糕', '鳳梨苦瓜湯', '高麗菜鹹粥', '紅燒里肌', '醬拌豆干', '炒時蔬', '玉米湯', '雞肉飯', '筍絲豆皮湯', '絲瓜鹹粥', '滷銀斑魚', '滷豆支', '沙茶玉米炒肉末', '炒時蔬', '大白菜豆皮湯']


day_7_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-28",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-28",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-28",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-28",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]

day_6_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-27",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-27",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-27",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-27",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]

day_5_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-26",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-26",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-26",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-26",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]

day_4_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-25",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-25",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-25",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-25",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]

day_3_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-24",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-24",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-24",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-24",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]

day_2_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-23",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-23",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-23",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-04-23",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]

day_1_docs = []