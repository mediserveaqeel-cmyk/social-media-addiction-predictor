import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')
enc = joblib.load('encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')
target_cols = joblib.load('target_cols.pkl')

st.title("Social Media Addiction Level Predictor")
st.write("Enter the details below to predict the addiction level class.")

# --- Categorical inputs (must match training categories) ---
gender = st.selectbox("Gender", enc.categories_[0])
occupation = st.selectbox("Occupation", enc.categories_[1])
relationship_status = st.selectbox("Relationship Status", enc.categories_[2])
primary_platform = st.selectbox("Primary Platform", enc.categories_[3])
late_night_usage = st.selectbox("Late Night Usage", enc.categories_[4])
first_check_morning = st.selectbox("First Check Morning", enc.categories_[5])
tried_to_cut_back = st.selectbox("Tried To Cut Back", enc.categories_[6])
failed_to_cut_back = st.selectbox("Failed To Cut Back", enc.categories_[7])

# --- Numeric inputs (edit these to match YOUR actual numeric columns) ---
age = st.number_input("Age", min_value=10, max_value=100, value=25)
daily_usage_hours = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=3.0)

if st.button("Predict"):
    # Build a single-row dataframe matching the categorical columns used in training
    cat_input = pd.DataFrame([{
        'Gender': gender,
        'Occupation': occupation,
        'Relationship_Status': relationship_status,
        'Primary_Platform': primary_platform,
        'Late_Night_Usage': late_night_usage,
        'First_Check_Morning': first_check_morning,
        'Tried_To_Cut_Back': tried_to_cut_back,
        'Failed_To_Cut_Back': failed_to_cut_back,
        # Addiction_Level is the target, not an input — encoder still expects
        # a placeholder column here since it was fit on all 9 columns together.
        'Addiction_Level': enc.categories_[8][0]
    }])

    encoded = enc.transform(cat_input)

    # Add numeric columns
    encoded['Age'] = age
    encoded['Daily_Usage_Hours'] = daily_usage_hours

    # Drop the Addiction_Level one-hot columns (they were only needed as a
    # placeholder to satisfy the encoder, not as real input features)
    encoded = encoded.drop(columns=[c for c in encoded.columns if c.startswith('Addiction_Level_')])

    # Reindex to match the exact feature order/columns used in training
    encoded = encoded.reindex(columns=feature_cols, fill_value=0)

    pred_idx = model.predict(encoded)[0]
    pred_label = target_cols[pred_idx]

    st.success(f"Predicted class: {pred_label}")
