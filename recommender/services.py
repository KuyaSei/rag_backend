import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import re

from rag.services.generator import ask_llm
from rag.services.patient_docs_retriever import get_patient_food_intake


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
# Dates




# ==================================
#
# ==================================

# ==================================
# Food Intake Calculation
# ==================================

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


# FORMAT AGGREGATED TOTAL INTAKE RESULTS
def format_calculated_intakes(aggregated_total_intake):
    return ", ".join(
        f"{volume} ml of {food}"
        for food, volume in aggregated_total_intake.items()
    )

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


# Returns a list of meal names available in the meals_meal table in the food_intakes_db database.
def get_list_of_meals():
    try:
        engine = create_engine("mysql+pymysql://root:Root%401234@localhost:3306/food_intakes_db")
        query = "SELECT * FROM meals_meal"
        meals_table = pd.read_sql(query, engine)

        meals_list = meals_table[['id', 'meal_name']].to_dict(orient='records')
        meal_names = [meal["meal_name"] for meal in meals_list]

        return meal_names

    except Exception as e:
        print("Error:", e)
        return []



# ==================================
# Get Nutritional Content Calculation from LLM
# ==================================
import requests
import json
import re

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


def get_nutritional_content_in_json(formatted_intakes):
    # [Exception Case]  formatted_intakes == None
    if formatted_intakes == None:
        return {"calories_kcal": 0, "protein_g": 0, "fats_g": 0, "carbohydrates_g": 0, "fiber_g": 0}

    prompt = f"""Calculate the total calories, protein, fats, and carbohydrates in {formatted_intakes}.

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


# Get patient DRIs' min and max range
def get_dri_min_max(dri_value):   
    min_val = round(dri_value * 0.8, 2)
    max_val = round(dri_value * 1.2, 2)

    return {
        "min": min_val,
        "max": max_val
    }


# Get nutrition remarks
def get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient):
    total = round(total_nutri_content[nutrient], 2)

    dri_range = recommended_intakes[nutrient]
    min_val = dri_range["min"]
    max_val = dri_range["max"]

    if total == 0:
        return "No intake"
    elif total < min_val:
        return "Below recommended"
    elif min_val <= total <= max_val:
        return "Meets recommended"
    else:
        return "Above recommended"


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