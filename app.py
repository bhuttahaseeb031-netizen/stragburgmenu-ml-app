# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set Streamlit Page Title
st.title("Starbucks Nutritional Analysis & Calorie Prediction App")

# Define file paths (Aap apni CSV files ko bhi GitHub repo mein upload karein)
expanded_menu_path = 'starbucks_drinkMenu_expanded.csv'
nutrition_drinks_path = 'starbucks-menu-nutrition-drinks.csv'
nutrition_food_path = 'starbucks-menu-nutrition-food.csv'

# Load datasets with correct encoding
try:
    df_expanded = pd.read_csv(expanded_menu_path)
    st.success("=== Expanded Drink Menu Dataset Loaded Successfully ===")
except Exception as e:
    st.error(f"Error loading expanded drink menu: {e}")
    df_expanded = None

# --- Clean Expanded Drink Dataset for Machine Learning ---
if df_expanded is not None:
    # Clean column names by stripping extra spaces
    df_expanded.columns = df_expanded.columns.str.strip()
    
    # Convert columns with string representations of numbers to numeric
    cols_to_numeric = ['Total Fat (g)', 'Vitamin A (% DV)', 'Vitamin C (% DV)', 'Calcium (% DV)', 'Iron (% DV)', 'Caffeine (mg)']
    for col in cols_to_numeric:
        df_expanded[col] = df_expanded[col].astype(str).str.replace('%', '').str.strip()
        df_expanded[col] = pd.to_numeric(df_expanded[col], errors='coerce')
        
    # Impute missing values with median
    df_expanded = df_expanded.fillna(df_expanded.median(numeric_only=True))
    
    # Define Features and Target
    features = ['Total Fat (g)', 'Trans Fat (g)', 'Saturated Fat (g)', 'Sodium (mg)', 'Total Carbohydrates (g)', 'Cholesterol (mg)', 'Dietary Fibre (g)', 'Sugars (g)', 'Protein (g)']
    X = df_expanded[features]
    y = df_expanded['Calories']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build Machine Learning Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions & Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Machine Learning Model Results (Predicting Calories)")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f} kcal")
    st.write(f"**R-squared Score:** {r2:.4f}")
    
    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")
    
    # Create a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graph 1: Actual vs. Predicted Calories
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='teal', edgecolor='w', s=80, ax=axes[0])
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_title('Machine Learning Model: Actual vs. Predicted Calories', fontsize=14, pad=10)
    axes[0].set_xlabel('Actual Calories (kcal)', fontsize=12)
    axes[0].set_ylabel('Predicted Calories (kcal)', fontsize=12)
    axes[0].legend(fontsize=11)
    
    # Graph 2: Feature Importances
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices] * 100
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis', hue=sorted_features, legend=False, ax=axes[1])
    axes[1].set_title('What Drives Calories? (Feature Importances)', fontsize=14, pad=10)
    axes[1].set_xlabel('Relative Importance (%)', fontsize=12)
    axes[1].set_ylabel('Nutritional Feature', fontsize=12)
    plt.tight_layout()
    
    # Show plots in Streamlit using st.pyplot instead of plt.show()
    st.pyplot(fig)
    
    # Decision Tree Model
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    dt_y_pred = dt_model.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_y_pred))
    dt_r2 = r2_score(y_test, dt_y_pred)
    
    st.subheader("Decision Tree Model Results")
    st.write(f"**Root Mean Squared Error (RMSE):** {dt_rmse:.2f} kcal")
    st.write(f"**R-squared Score:** {dt_r2:.4f}")
