import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import re

from rag.services.generator import ask_llm
from rag.services.patient_docs_retriever import get_patient_food_intake

# OKAY - Basic Patient Data will be used for every active view
# Get patient info 
def get_patient_info(patient):
    patient_sex = patient.get("sex")
    patient_age = patient.get("age")
    patient_height_cm = patient.get("height_cm")
    patient_weight_kg = patient.get("weight_kg")
    patient_bmi = patient.get("bmi")
    patient_activity_level = patient.get("activity_level")

    patient_info = (
        f"The LTC patient is a {patient_age}-year-old {patient_sex} with a height of {patient_height_cm} cm and a weight of {patient_weight_kg} kg, resulting in a BMI of {patient_bmi}. "
        f"The LTC patient has a {patient_activity_level} physical activity level."
    )
    return patient_info

# STANDBY only used by MonthlyPatientFoodIntakeRecommenderView
# Dates
def get_current_month():
    """Return the current month as an integer (1–12)."""
    return datetime.now(ZoneInfo("Asia/Taipei")).month

def get_current_year():
    """Return the current year as a 4-digit integer."""
    return datetime.now(ZoneInfo("Asia/Taipei")).year

def get_current_month_str():
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%m")

def get_current_year_str():
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y")

def get_current_month_name():
    curdate = datetime.now(ZoneInfo("Asia/Taipei"))
    return curdate.strftime("%B")

# STANDBY - generates list of all days in the month. USed by standby only View
def get_dates_in_current_month():
    """Return a list of all dates in the current month in YYYY-MM-DD format."""
    curdate = datetime.now(ZoneInfo("Asia/Taipei"))
    year = curdate.year
    month = curdate.month

    days_in_month = calendar.monthrange(year, month)[1]

    return [
        f"{year}-{month:02d}-{day:02d}"
        for day in range(1, days_in_month + 1)
    ]

# STANDBY - Iterates through every day in the month to fetch Qdrant data. Used only by the standby Monthly view.
def get_food_intake_results_in_curmonth(dates_in_current_month, pid):
    monthly_results = []

    for date in dates_in_current_month:
        fi_results = get_patient_food_intake(pid, date)

        monthly_results.append({
            "date": date,
            "food_intakes": fi_results if fi_results else "Missing"
        })

    return monthly_results

# STANDBY - Formats the massive monthly data pull into a string for the LLM. Used only by the standby Monthly view.
def create_monthly_food_intake_context(fi_results_in_curmonth_list, date_format="month_day"):
    monthly_context = []

    for day_result in fi_results_in_curmonth_list:
        date_str = day_result["date"]
        date_obj = datetime.fromisoformat(date_str)
        
        # Format date
        if date_format == "month_day":
            formatted_date = date_obj.strftime("%B %-d")  # March 1
        else:
            formatted_date = date_obj.strftime("%Y-%m-%d")  # 2026-03-01
        
        food_intakes = day_result["food_intakes"]
        day_context = f"{formatted_date}:\n"

        if food_intakes == "Missing":
            day_context += "- Missing"
        else:
            for doc, metadata in food_intakes:
                day_context += f"- {doc}\n"

        monthly_context.append(day_context.strip())

    return "\n\n".join(monthly_context)



# ==================================
# Food Intake Calculation
# ==================================

# OKAY - core math engine. takes before meal array and subtracts after meal array to calculate how much ml of each food the patient has consumed. 
def calculate_food_item_intake(food_intake_docs, debug=False):
    """
    Calculate food intake per item in volume (mL) based on before and after meal data. 
    Aggregate calculated food volume intake per item.
    """

    # 1. Organize food intake ,first, by meal_time (lunch/dinner) and ,second, by meal_phase (before/after)

    # OUTPUT DATA
    # {'lunch': { 
    #              'before': {'broccoli': 24.305, 'chicken': 29.2051, 'rice': 319.5655}, 
    #              'after': {'cabbage': 30.4233, 'chicken': 36.1196, 'rice': 289.3211}
    #          },
    # 'dinner': { 
    #              'before': {'broccoli': 24.305, 'chicken': 29.2051, 'rice': 319.5655}, 
    #              'after': {'cabbage': 30.4233, 'chicken': 36.1196, 'rice': 289.3211}
    #          }
    # }

    meals = {} 

    for doc in food_intake_docs:
        metadata = doc.get("metadata", {})
        meal_time = metadata.get("meal_time_en")
        meal_phase = metadata.get("meal_phase_en")
        food_items = metadata.get("food_items", [])

        if meal_time not in meals:
            meals[meal_time] = {"before": {}, "after": {}}

        for item in food_items:
            food_class_name = item.get("food_class")
            volume_ml = item.get("volume_ml", 0)
            
            # CASE 1: Multiple instances of the same food class (e.g., chicken_1, chicken_2) → aggregate volume under the same food class name (e.g., chicken)
            # [For multi-instance food_class handling] Normalize food_class_name by removing numeric suffixes (e.g., chicken_1 -> chicken)
            normalized_name = re.sub(r'_\d+$', '', food_class_name)

            # Aggregate volume
            if normalized_name in meals[meal_time][meal_phase]:
                meals[meal_time][meal_phase][normalized_name] += volume_ml
            else:
                meals[meal_time][meal_phase][normalized_name] = volume_ml

    if debug:
        print("[STEP 1] Organize data")
        print("Meals structure:", meals)


    # 2. Compute intake per food item (before - after)

    # OUTPUT DATA
    # By meal: {'dinner': {'broccoli': 24.305, 'rice': 30.244399999999985, 'chicken': 0}}
    # Aggregated total intake: {'broccoli': 24.305, 'rice': 30.244399999999985, 'chicken': 0}
    
    by_meal_result = {}
    total_intake = {}

    for meal_time, meal_phases in meals.items():
        before_items = meal_phases.get("before", {})
        after_items = meal_phases.get("after", {})

        # [For no after meal_phase case handling] If after meal data is missing, assume all before meal volume is consumed (i.e., after meal volume = 0)
        # CASE 2: No AFTER meal phase → everything consumed
        if not after_items:
            after_items = {food: 0 for food in before_items}

        # [For new food items in after meal_phase handling] If after meal has detected food items not present in before meal, assume they were not part of the original meal and thus not consumed. Therefore, disregard new items in after meal for intake calculation)
        # CASE 3: AFTER exists but has new food items not in BEFORE → disregard new items 
        after_items = {
            food: vol for food, vol in after_items.items()
            if food in before_items
        }

        # CASE 4: AFTER exists but missing some foods → assume fully consumed
        for food in before_items:
            if food not in after_items:
                after_items[food] = 0

        all_foods = set(before_items.keys()) # Creates a set of all food items to be processed for intake calculation.
        meal_result = {}

        for food in all_foods:
            before_val = before_items.get(food, 0)
            after_val = after_items.get(food, 0)
            intake = before_val - after_val

            # Prevent negative intake (optional safety)
            intake = max(intake, 0)

            meal_result[food] = intake

            # Aggregate total intake across meals
            total_intake[food] = total_intake.get(food, 0) + intake

        by_meal_result[meal_time] = meal_result

    if debug:
        print("\n[STEP 2] Intake calculation")
        print("By meal:", by_meal_result)
        print("Aggregated total intake:", total_intake)


    # 3. Format total intake per food item across all meals

    # OUTPUT DATA
    # {'by_meal': 
    #     {
    #         'dinner': {'broccoli': 24.305, 'rice': 30.244399999999985, 'chicken': 0}
    #     }, 
    # 'aggregated_total': 
    #     {
    #         'broccoli': 24.3, 'rice': 30.2, 'chicken': 0
    #     }
    # }

    total_result = {
        food: round(volume_ml, 1) 
        for food, volume_ml in total_intake.items()
    }

    res = {
        "by_meal": by_meal_result,
        "aggregated_total": total_result
    }
    
    if debug:
        print("\n[STEP 3] Formatted Results")
        print(res)

    return res

# OKAY - formats math output into a readable string (e.g., 30 ml of rice, 20 ml of chicken) for the LLM to use. 
# FORMAT AGGREGATED TOTAL INTAKE RESULTS
def format_calculated_intakes(aggregated_total_intake):
    return ", ".join(
        f"{volume} ml of {food}"
        for food, volume in aggregated_total_intake.items()
    )

# OKAY - same above formats the data slightly diffeently for the final JSON response
def format_calculated_intakes_for_response(aggregated_total_intake):
    return ", ".join(
        f"{food} ({volume} ml)"
        for food, volume in aggregated_total_intake.items()
    )





# ==================================
# Get Meals from Database
# ==================================
import pymysql
import pandas as pd
from sqlalchemy import create_engine


# OKAY - queries the database to get master list of available meal to give the LLM options to recommend from

# Returns a list of meal names available in the meals_meal table in the food_intakes_db database.
def get_list_of_meals():
    # Connect to the database
    engine = create_engine("mysql+pymysql://root:root@localhost:3306/food_intakes_db")

    # Query the meals table
    query = "SELECT * FROM meals_meal"

    meals_table = pd.read_sql(query, engine)
    meals_list = meals_table[['id', 'meal_name']].to_dict(orient='records')
    meal_names = [meal["meal_name"] for meal in meals_list]

    return meal_names



# ==================================
# Get Nutritional Content Calculation from LLM
# ==================================
import requests
import json
import re

# OKAY - strips the formatting so python can read actual JSON object
def preprocess_llm_response(raw):
    # 1. remove markdown if any
    raw = re.sub(r"```json|```", "", raw)

    # 2. fix double braces
    raw = raw.replace("{{", "{").replace("}}", "}")

    # 3. remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # 4. strip whitespace
    raw = raw.strip()

    return json.loads(raw)

# OKAY / STANDBY - sends the formatted food intake string to the LLM with a prompt to calculate the total nutritional content (calories, protein, fats, carbohydrates, fiber) in the patient's food intake. The LLM responds with a JSON object containing the calculated nutritional content.
def get_nutritional_content_in_json(formatted_intakes):
    # [Exception Case]  formatted_intakes == None
    if formatted_intakes == None:
        return {"calories_kcal": 0, "protein_g": 0, "fats_g": 0, "carbohydrates_g": 0, "fiber_g": 0}

    prompt = f"""Calculate the total calories, protein, fats, carbohydrates and fiber in {formatted_intakes}.

Return answers ONLY in JSON in this EXACT FORMAT:
{{
    "calories_kcal": number,
    "protein_g": number,
    "fats_g": number,
    "carbohydrates_g": number,
    "fiber_g": number,
}}
"""

    raw = ask_llm(prompt)
    preprocessed_res = preprocess_llm_response(raw)

    return preprocessed_res


# OKAY - compares what the patient actually ate against their DRI target and gives remarks (below, meets, or above recommended)

# Get nutrition remarks. (As of now, consider +1-10% of nutrient still acceptable)
def get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient):
    total = round(total_nutri_content[nutrient], 2)
    recommended = round(recommended_intakes[nutrient], 2)
    
    # Create a ±20% tolerance range around the target
    lower_bound = recommended * 0.80
    upper_bound = recommended * 1.20

    if total == 0:
        intake_remarks = "No intake"
    elif total < lower_bound:
        intake_remarks = "Below recommended"
    elif total <= upper_bound:
        intake_remarks = "Meets recommended"
    else:
        intake_remarks = "Above recommended"

    return intake_remarks


# ==================================
# Meal Recommendations
# ==================================
nutrient_labels = {
    "calories_kcal": "calories",
    "protein_g": "protein",
    "fats_g": "fats",
    "carbohydrates_g": "carbohydrates",
    "fiber_g": "fiber"
}


# OKay - A helper function that takes the remarks generated above and sorts the nutrients into three clean lists: deficient, excessive, and normal.
def categorize_nutrients(nutrition_remarks):
    deficient = []
    excessive = []
    normal = []

    for nutrient, remark in nutrition_remarks.items():
        if remark in ["No intake", "Below recommended"]:
            deficient.append(nutrient)
        elif remark == "Above recommended":
            excessive.append(nutrient)
        else:
            normal.append(nutrient)
    return deficient, excessive, normal


# OBSOLETE - old Single-Retriever LLM prompt. It ignores the HPA database entirely and uses hardcoded IF statements. Because we built the master prompt directly inside views.py

# def get_daily_meal_recommendations(meal_names, nutrition_remarks):
#     deficient, excessive, _ = categorize_nutrients(nutrition_remarks)

#     if not deficient and not excessive:
#         return "All nutrients are within recommended levels."

#     messages = []

#     # Deficient nutrients (No intake/Below recommended)
#     for nutrient in deficient:
#         readable = nutrient_labels[nutrient]
#         remark = nutrition_remarks[nutrient]

#         if remark == "No intake":
#             messages.append(f"Patient has no {readable} intake.")
#         else:
#             messages.append(f"{readable} is below recommended value.")

#     # Excess nutrients (Above recommended)
#     for nutrient in excessive:
#         readable = nutrient_labels[nutrient]
#         messages.append(f"{readable} is above recommended value and should be moderated.")

#     condition_text = " ".join(messages)
#     meals_text = ", ".join(meal_names)

#     prompt = f"""
# You are a clinical nutrition assistant.

# {condition_text}

# Here are available meals:
# {meals_text}

# Task:
# 1. For nutrients that are below recommended or no intake:
#     - Add to response "Here are some meal recommendations high in <NUTRIENT>"
#    - Recommend meals high in those nutrients.
# 2. For nutrients that are above recommended:
#     - Add to response "Here are some lighter meal options"
#    - Recommend meals low in those nutrients OR lighter options.

# Rules:
# - Only choose from the given meal list
# - Give 5 recommendations per nutrient in the format: meal_1, meal_2, meal_3, meal_4, meal_5
# - Keep response concise
# - Say nutrient and condition before giving meal recommendations
# """

#     response = ask_llm(prompt)
#     return response



# OBSOLETE - same logic as above but for the weekly view. Because we built the master prompt directly inside views.py, this function is currently bypassed by the frontend.
# def get_weekly_meal_recommendations(meal_names, nutrition_remarks):
#     deficient, excessive, _ = categorize_nutrients(nutrition_remarks)

#     if not deficient and not excessive:
#         return "All nutrients are within recommended levels."

#     messages = []

#     # Deficient nutrients (Below recommended)
#     for nutrient in deficient:
#         readable = nutrient_labels[nutrient]
#         messages.append(f"{readable} is below recommended value this week.")

#     # Excess nutrients (Above recommended)
#     for nutrient in excessive:
#         readable = nutrient_labels[nutrient]
#         messages.append(f"{readable} is above recommended value this week and should be moderated.")

#     condition_text = " ".join(messages)
#     meals_text = ", ".join(meal_names)

#     prompt = f"""
# You are a clinical nutrition assistant.

# {condition_text}

# Here are available meals:
# {meals_text}

# Task:
# 1. For nutrients that are below recommended or no intake:
#     - Add to response "Here are some meal recommendations high in <NUTRIENT>"
#    - Recommend meals high in those nutrients.
# 2. For nutrients that are above recommended:
#     - Add to response "Here are some lighter meal options"
#    - Recommend meals low in those nutrients OR lighter options.

# Rules:
# - Only choose from the given meal list
# - Give 5 recommendations per nutrient in the format: meal_1, meal_2, meal_3, meal_4, meal_5
# - Keep response concise
# - Say nutrient and condition before giving meal recommendations
# """

#     response = ask_llm(prompt)
#     return response


# OBSOLETE - same logic as above but for the monthly view. Because we built the master prompt directly inside views.py, this function is currently bypassed by the frontend.
# def get_monthly_meal_recommendations(meal_names, nutrition_remarks):
#     deficient, excessive, _ = categorize_nutrients(nutrition_remarks)

#     if not deficient and not excessive:
#         return "All nutrients are within recommended levels."

#     messages = []

#     # Deficient nutrients (Below recommended)
#     for nutrient in deficient:
#         readable = nutrient_labels[nutrient]
#         messages.append(f"{readable} is below recommended value this month.")

#     # Excess nutrients (Above recommended)
#     for nutrient in excessive:
#         readable = nutrient_labels[nutrient]
#         messages.append(f"{readable} is above recommended value this month and should be moderated.")

#     condition_text = " ".join(messages)
#     meals_text = ", ".join(meal_names)

#     prompt = f"""
# You are a clinical nutrition assistant.

# {condition_text}

# Here are available meals:
# {meals_text}

# Task:
# 1. For nutrients that are below recommended or no intake:
#     - Add to response "Here are some meal recommendations high in <NUTRIENT>"
#    - Recommend meals high in those nutrients.
# 2. For nutrients that are above recommended:
#     - Add to response "Here are some lighter meal options"
#    - Recommend meals low in those nutrients OR lighter options.

# Rules:
# - Only choose from the given meal list
# - Give 5 recommendations per nutrient in the format: meal_1, meal_2, meal_3, meal_4, meal_5
# - Keep response concise
# - Say nutrient and condition before giving meal recommendations
# """

#     response = ask_llm(prompt)
#     return response