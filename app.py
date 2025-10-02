import os

# If models file doesn't exist, train models
if not os.path.exists('diabetes_models.pkl'):
    st.warning("Training models... This will take 30 seconds on first deployment.")
    # Add code here to train and save models
    # (copy the training code from your notebook)

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 0rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    with open('diabetes_models.pkl', 'rb') as file:
        data = pickle.load(file)
    return data['models'], data['scaler'], data['results_df']

try:
    models, scaler, results_df = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Error loading models: {e}")

# Title
st.title("🏥 Diabetes Prediction System - Model Comparison")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔍 Make Prediction", "📈 About the Project"])

# TAB 1: MODEL COMPARISON
with tab1:
    if models_loaded:
        st.header("Model Performance Comparison")
        
        # Display comparison table
        st.subheader("Performance Metrics")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for all metrics
            fig = px.bar(results_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        title="All Metrics Comparison",
                        barmode='group',
                        height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Focus on Recall (most important for medical)
            fig = px.bar(results_df, x='Model', y='Recall',
                        title="Recall Comparison (Most Important for Medical Screening)",
                        color='Recall',
                        color_continuous_scale='RdYlGn',
                        height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        best_model = results_df.iloc[0]['Model']
        best_recall = results_df['Recall'].max()
        best_recall_model = results_df.loc[results_df['Recall'].idxmax(), 'Model']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Overall Model", best_model, results_df.iloc[0]['F1-Score'])
        col2.metric("Best Recall Model", best_recall_model, f"{best_recall:.3f}")
        col3.metric("Avg Performance", "All Models", f"{results_df['Accuracy'].mean():.3f}")

# TAB 2: PREDICTION
with tab2:
    if models_loaded:
        st.header("Make a Prediction")
        
        # Model selector
        selected_model_name = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            help="Select which ML algorithm to use for prediction"
        )
        
        # Show model performance
        model_perf = results_df[results_df['Model'] == selected_model_name].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{model_perf['Accuracy']:.3f}")
        col2.metric("Precision", f"{model_perf['Precision']:.3f}")
        col3.metric("Recall", f"{model_perf['Recall']:.3f}")
        col4.metric("F1-Score", f"{model_perf['F1-Score']:.3f}")
        
        st.markdown("---")
        
        # Input section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            pregnancies = st.number_input("Pregnancies", 0, 17, 1)
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        
        with col2:
            st.subheader("Additional Metrics")
            insulin = st.slider("Insulin (µU/mL)", 0, 846, 80)
            bmi = st.number_input("BMI", 0.0, 67.1, 25.0, 0.1)
            dpf = st.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.47, 0.001)
            age = st.slider("Age", 21, 81, 33)
        
        if st.button("🔍 Predict with Selected Model", type="primary"):
            # Prepare input
            input_array = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_array)
            
            # Get selected model
            selected_model = models[selected_model_name]
            
            # Predict
            prediction = selected_model.predict(input_scaled)[0]
            probability = selected_model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("### ⚠️ HIGH RISK")
                else:
                    st.success("### ✅ LOW RISK")
            
            with col2:
                st.metric("Diabetes Probability", f"{probability[1]*100:.1f}%")
            
            with col3:
                st.metric("Model Used", selected_model_name)
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare with all models
            st.markdown("---")
            st.subheader("How would other models predict?")
            
            comparison_data = []
            for model_name, model in models.items():
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]
                comparison_data.append({
                    'Model': model_name,
                    'Prediction': 'Diabetic' if pred == 1 else 'Non-Diabetic',
                    'Probability': f"{prob*100:.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# TAB 3: ABOUT
with tab3:
    st.header("About This Project")
    
    st.markdown("""
    ### Project Overview
    
    **Based on tutorial by Siddhardhan:**
    - **Video:** "Project 2: Diabetes Prediction using Machine Learning with Python | End To End Python ML Project"
    - **Link:** [Watch on YouTube](https://www.youtube.com/watch?v=xUE7SjVx9bQ&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=2)
    
    The original tutorial used only SVM and didn't address data quality issues. I expanded it significantly with data cleaning, 5 model comparison, and this interactive web app.
    
    ---
    
    This diabetes prediction system was developed as a comprehensive machine learning project,
    starting from a basic tutorial and expanding it with advanced techniques.
    
    ### What I Built
    
    1. **Data Analysis & Cleaning**
       - Identified 652 medically impossible zero values (48% of rows affected)
       - Implemented class-aware median imputation
       - Created extensive visualizations (correlation matrix, distributions, boxplots)
    
    2. **Model Development**
       - Trained and compared 5 different algorithms
       - Focused on Recall metric (critical for medical screening)
       - Implemented ensemble methods
    
    3. **Interactive Application**
       - Built this Streamlit web app for easy model comparison
       - Real-time predictions with any selected model
       - Visual risk indicators and recommendations
    
    ### Key Learning
    
    **Initial approach:** Followed tutorial → 77% accuracy with SVM
    
    **Problem discovered:** Accuracy was misleading. The model had only 52% recall, 
    meaning it missed nearly half of diabetic patients!
    
    **Solution:** After data cleaning and comparing 5 algorithms, optimized for recall 
    (medical priority). Achieved 56% recall with Random Forest.
    
    **Reality check:** The modest improvement reveals the real limitation - 
    the PIMA dataset is small (768 patients) and dated (1990s). 
    
    In a real project, I would recommend:
    - Collecting more data
    - Adding modern biomarkers (HbA1c, fasting insulin, etc.)
    - Gathering family history details
    
    ### Model Performance Summary
    """)
    
    if models_loaded:
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### Technologies Used
    - Python, Pandas, NumPy, Scikit-learn
    - Matplotlib, Seaborn, Plotly
    - Streamlit for web deployment
    
    ### Dataset
    PIMA Indians Diabetes Database (768 patients, 8 features)
    
    ⚠️ **Disclaimer:** This is an educational project, not for actual medical diagnosis.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Developed by Willen CHIBOUT | Machine Learning Portfolio Project</p>
    <p>Data: PIMA Indians Diabetes Database</p>
</div>
""", unsafe_allow_html=True)
