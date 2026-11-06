import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Starbucks Menu & Nutrition Predictor App")

# --- Define file paths (Assuming files are in the same repository folder) ---
expanded_menu_path = 'starbucks_drinkMenu_expanded.csv'
nutrition_drinks_path = 'starbucks-menu-nutrition-drinks.csv'
nutrition_food_path = 'starbucks-menu-nutrition-food.csv'

# --- Load datasets with correct encoding ---
@st.cache_data
def load_data():
    try:
        df_expanded = pd.read_csv(expanded_menu_path)
    except Exception as e:
        df_expanded = None

    try:
        df_drinks = pd.read_csv(nutrition_drinks_path)
    except Exception as e:
        df_drinks = None

    try:
        df_food = pd.read_csv(nutrition_food_path, encoding='utf-16')
    except Exception as e:
        try:
            df_food = pd.read_csv(nutrition_food_path, encoding='latin1')
        except Exception as e2:
            df_food = None
            
    return df_expanded, df_drinks, df_food

df_expanded, df_drinks, df_food = load_data()

if df_expanded is not None:
    st.success("=== Expanded Drink Menu Dataset Loaded Successfully ===")
    
    # Clean column names by stripping extra spaces
    df_expanded.columns = df_expanded.columns.str.strip()
    
    # Convert columns with string representations of numbers to numeric
    cols_to_numeric = ['Total Fat (g)', 'Vitamin A (% DV)', 'Vitamin C (% DV)', 'Calcium (% DV)', 'Iron (% DV)', 'Caffeine (mg)']
    for col in cols_to_numeric:
        if col in df_expanded.columns:
            df_expanded[col] = df_expanded[col].astype(str).str.replace('%', '').str.strip()
            df_expanded[col] = pd.to_numeric(df_expanded[col], errors='coerce')
            
    st.write("Dataset Preview:", df_expanded.head())
else:
    st.error("Could not load `starbucks_drinkMenu_expanded.csv`. Please ensure it is uploaded in the repository.")
