import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ──────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .main { padding-top: 1rem; }
        h1 { color: #1f2937; font-weight: 700; }
        .subtitle { color: #6b7280; font-size: 1.05rem; margin-bottom: 1.5rem; }
        .metric-card {
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            text-align: center;
        }
        .metric-label { color: #6b7280; font-size: 0.85rem; }
        .metric-value { color: #111827; font-size: 1.6rem; font-weight: 700; }
        .stButton>button {
            width: 100%;
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            padding: 0.6rem 0;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover { background-color: #1d4ed8; }
        .result-card {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 1rem;
        }
        .result-label { color: #1e40af; font-size: 1rem; font-weight: 600; }
        .result-value { color: #1e3a8a; font-size: 2rem; font-weight: 800; margin-top: 0.25rem; }
    </style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Load model artifacts
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    enc = joblib.load('encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    target_cols = joblib.load('target_cols.pkl')
    try:
        accuracy = joblib.load('accuracy.pkl')
    except FileNotFoundError:
        accuracy = None
    return model, enc, feature_cols, target_cols, accuracy

model, enc, feature_cols, target_cols, accuracy = load_artifacts()


# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.title("📱 Social Media Addiction Level Predictor")
st.markdown(
    '<div class="subtitle">A multiclass logistic regression model that estimates addiction '
    'severity from usage habits, psychological scores, and lifestyle factors.<br>'
    'Developed by <b>Muhammad Aqeel</b></div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Model Type</div>'
        f'<div class="metric-value" style="font-size:1.1rem;">Logistic Regression</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    acc_display = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Test Accuracy</div>'
        f'<div class="metric-value">{acc_display}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Prediction Classes</div>'
        f'<div class="metric-value">{len(target_cols)}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()


# ──────────────────────────────────────────────────────────────
# Sidebar — categorical inputs
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Profile")
    gender = st.selectbox("Gender", enc.categories_[0])
    occupation = st.selectbox("Occupation", enc.categories_[1])
    relationship_status = st.selectbox("Relationship Status", enc.categories_[2])
    primary_platform = st.selectbox("Primary Platform", enc.categories_[3])

    st.header("🌙 Habits")
    late_night_usage = st.selectbox("Late Night Usage", enc.categories_[4])
    first_check_morning = st.selectbox("First Check Morning", enc.categories_[5])
    tried_to_cut_back = st.selectbox("Tried To Cut Back", enc.categories_[6])
    failed_to_cut_back = st.selectbox("Failed To Cut Back", enc.categories_[7])


# ──────────────────────────────────────────────────────────────
# Main form — numeric inputs, grouped in tabs
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Usage & Behavior", "🧠 Psychological Scores", "🛌 Lifestyle"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
        daily_usage_hours = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=3.0)
    with c2:
        platforms_used_count = st.number_input("Platforms Used Count", min_value=0, max_value=20, value=3)
        posts_per_week = st.number_input("Posts Per Week", min_value=0, max_value=200, value=5)
    with c3:
        notifications_per_day = st.number_input("Notifications Per Day", min_value=0, max_value=500, value=20)
        scroll_without_purpose = st.number_input("Scroll Without Purpose (score)", min_value=0.0, max_value=10.0, value=5.0)

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        fomo_score = st.number_input("FOMO Score", min_value=0.0, max_value=10.0, value=5.0)
        anxiety_score = st.number_input("Anxiety Score", min_value=0.0, max_value=10.0, value=5.0)
    with c2:
        social_comparison_score = st.number_input("Social Comparison Score", min_value=0.0, max_value=10.0, value=5.0)
        depression_score = st.number_input("Depression Score", min_value=0.0, max_value=10.0, value=5.0)
    with c3:
        validation_seeking_score = st.number_input("Validation Seeking Score", min_value=0.0, max_value=10.0, value=5.0)
        loneliness_score = st.number_input("Loneliness Score", min_value=0.0, max_value=10.0, value=5.0)
    self_esteem_score = st.number_input("Self Esteem Score", min_value=0.0, max_value=10.0, value=5.0)

with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        sleep_quality_score = st.number_input("Sleep Quality Score", min_value=0.0, max_value=10.0, value=5.0)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    with c2:
        productivity_loss_score = st.number_input("Productivity Loss Score", min_value=0.0, max_value=10.0, value=5.0)
        offline_relationship_quality = st.number_input("Offline Relationship Quality", min_value=0.0, max_value=10.0, value=5.0)
    with c3:
        physical_activity_hrs_week = st.number_input("Physical Activity Hrs/Week", min_value=0.0, max_value=50.0, value=3.0)
        screen_free_time_hrs = st.number_input("Screen Free Time Hrs", min_value=0.0, max_value=24.0, value=2.0)
    mental_wellbeing_score = st.number_input("Mental Wellbeing Score", min_value=0.0, max_value=10.0, value=5.0)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Addiction Level")


# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────
if predict_clicked:
    cat_input = pd.DataFrame([{
        'Gender': gender,
        'Occupation': occupation,
        'Relationship_Status': relationship_status,
        'Primary_Platform': primary_platform,
        'Late_Night_Usage': late_night_usage,
        'First_Check_Morning': first_check_morning,
        'Tried_To_Cut_Back': tried_to_cut_back,
        'Failed_To_Cut_Back': failed_to_cut_back,
        'Addiction_Level': enc.categories_[8][0],  # placeholder only, dropped below
    }])

    encoded = enc.transform(cat_input)
    encoded = encoded.drop(columns=[c for c in encoded.columns if c.startswith('Addiction_Level_')])

    numeric_values = {
        'Age': age,
        'Daily_Usage_Hours': daily_usage_hours,
        'Platforms_Used_Count': platforms_used_count,
        'Posts_Per_Week': posts_per_week,
        'Notifications_Per_Day': notifications_per_day,
        'FOMO_Score': fomo_score,
        'Social_Comparison_Score': social_comparison_score,
        'Validation_Seeking_Score': validation_seeking_score,
        'Scroll_Without_Purpose': scroll_without_purpose,
        'Anxiety_Score': anxiety_score,
        'Depression_Score': depression_score,
        'Loneliness_Score': loneliness_score,
        'Self_Esteem_Score': self_esteem_score,
        'Sleep_Quality_Score': sleep_quality_score,
        'Sleep_Hours': sleep_hours,
        'Productivity_Loss_Score': productivity_loss_score,
        'Offline_Relationship_Quality': offline_relationship_quality,
        'Physical_Activity_Hrs_Week': physical_activity_hrs_week,
        'Screen_Free_Time_Hrs': screen_free_time_hrs,
        'Mental_Wellbeing_Score': mental_wellbeing_score,
    }
    for col, val in numeric_values.items():
        encoded[col] = val

    encoded = encoded.reindex(columns=feature_cols, fill_value=0)

    pred_idx = model.predict(encoded)[0]
    pred_label = target_cols[pred_idx].replace('Addiction_Level_', '')

    # Confidence, if the model supports it
    confidence_str = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(encoded)[0]
        confidence = proba[pred_idx] * 100
        confidence_str = f'<div style="color:#1e40af; margin-top:0.5rem;">Confidence: {confidence:.1f}%</div>'

    st.markdown(
        f'<div class="result-card">'
        f'<div class="result-label">Predicted Addiction Level</div>'
        f'<div class="result-value">{pred_label}</div>'
        f'{confidence_str}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Built by Muhammad Aqeel · Powered by scikit-learn & Streamlit · Predictions are estimates based on a logistic regression model, not clinical assessments.")
