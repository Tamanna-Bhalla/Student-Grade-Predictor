import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.data_processing import DataProcessor
from src.model import ModelTrainer
from src.utils import PredictionUtils, ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎓 Student Grade Predictor")
st.markdown("**Predict student performance using advanced machine learning models**")
st.markdown("---")

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
    
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()
    
if 'prediction_utils' not in st.session_state:
    st.session_state.prediction_utils = PredictionUtils()

if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator()

# Check if models exist, if not train them
@st.cache_resource
def initialize_models():
    """Initialize and train models if they don't exist"""
    model_trainer = ModelTrainer()
    
    # Check if models exist
    rf_reg_path = "models/random_forest_regression.joblib"
    rf_clf_path = "models/random_forest_classification.joblib"
    xgb_reg_path = "models/xgboost_regression.joblib"
    xgb_clf_path = "models/xgboost_classification.joblib"
    
    models_exist = all(os.path.exists(path) for path in [rf_reg_path, rf_clf_path, xgb_reg_path, xgb_clf_path])
    
    if not models_exist:
        st.info("🔄 Training machine learning models... This may take a few minutes.")
        with st.spinner("Training models..."):
            try:
                model_trainer.train_all_models()
                st.success("✅ Models trained successfully!")
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
                return False
    
    return True

# Initialize models
models_ready = initialize_models()

if not models_ready:
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "📊 Data Overview", "🎯 Make Prediction", "📈 Model Performance"]
)

if page == "🏠 Home":
    # Home page
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to Student Grade Predictor")
        st.markdown("""
        This application uses advanced machine learning algorithms to predict student academic performance
        based on comprehensive demographic, social, and academic factors.
        
        **Features:**
        - 🎯 **Accurate Predictions**: Uses RandomForest and XGBoost algorithms
        - 📊 **Feature Importance**: SHAP explanations for model interpretability
        - 📝 **Detailed Reports**: Generate comprehensive PDF/CSV reports
        - 💡 **Actionable Insights**: Personalized recommendations for improvement
        
        **How it works:**
        1. Input student information across 33 different factors
        2. Our ML models analyze the data and predict academic performance
        3. Get explanations and recommendations based on the predictions
        """)
        
    with col2:
        st.info("""
        **Quick Stats:**
        - 33 input features
        - 2 ML algorithms
        - Regression & Classification
        - SHAP explanations
        - PDF/CSV reports
        """)

elif page == "📊 Data Overview":
    # Data overview page
    st.header("📊 Dataset Overview")
    
    try:
        # Load and display dataset info
        data_processor = st.session_state.data_processor
        df = data_processor.load_data()
        
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns) - 1)  # Excluding target
            with col3:
                st.metric("Avg Grade", f"{df['G3'].mean():.2f}")
            with col4:
                st.metric("Grade Range", f"{df['G3'].min()}-{df['G3'].max()}")
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head(), use_container_width=True)
            
            # Feature distributions
            st.subheader("Feature Distributions")
            
            # Categorical features
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if 'G3' in numerical_features:
                numerical_features.remove('G3')
            
            # Grade distribution
            fig = px.histogram(df, x='G3', nbins=20, title="Grade Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation with target
            correlations = df[numerical_features + ['G3']].corr()['G3'].drop('G3').abs().sort_values(ascending=False)
            
            fig = px.bar(
                x=correlations.values[:10],
                y=correlations.index[:10],
                orientation='h',
                title="Top 10 Features Correlated with Grades"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("❌ Could not load dataset. Please ensure 'data/student_data.csv' exists.")
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

elif page == "🎯 Make Prediction":
    # Prediction page
    st.header("🎯 Student Grade Prediction")
    
    # Input form
    st.subheader("📝 Student Information")
    
    # Create input form with organized sections
    with st.form("student_form"):
        # Section 1: Basic Information
        st.markdown("### 👤 Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            school = st.selectbox(
                "School Type",
                options=["GP", "MS"],
                help="GP - Gabriel Pereira or MS - Mousinho da Silveira"
            )
            sex = st.selectbox("Gender", options=["F", "M"])
            age = st.slider("Age", min_value=15, max_value=22, value=16)
        
        with col2:
            address = st.selectbox(
                "Address Type",
                options=["U", "R"],
                help="U - Urban, R - Rural"
            )
            famsize = st.selectbox(
                "Family Size",
                options=["LE3", "GT3"],
                help="LE3 - Less or equal to 3, GT3 - Greater than 3"
            )
            Pstatus = st.selectbox(
                "Parent's Cohabitation Status",
                options=["T", "A"],
                help="T - Living together, A - Apart"
            )
        
        with col3:
            Medu = st.slider(
                "Mother's Education",
                min_value=0, max_value=4, value=2,
                help="0 - None, 1 - Primary (4th grade), 2 - 5th-9th grade, 3 - Secondary, 4 - Higher education"
            )
            Fedu = st.slider(
                "Father's Education",
                min_value=0, max_value=4, value=2,
                help="0 - None, 1 - Primary (4th grade), 2 - 5th-9th grade, 3 - Secondary, 4 - Higher education"
            )
        
        # Section 2: Family Background
        st.markdown("### 👨‍👩‍👧‍👦 Family Background")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Mjob = st.selectbox(
                "Mother's Job",
                options=["teacher", "health", "services", "at_home", "other"]
            )
            Fjob = st.selectbox(
                "Father's Job",
                options=["teacher", "health", "services", "at_home", "other"]
            )
        
        with col2:
            reason = st.selectbox(
                "Reason for School Choice",
                options=["home", "reputation", "course", "other"]
            )
            guardian = st.selectbox(
                "Guardian",
                options=["mother", "father", "other"]
            )
        
        with col3:
            traveltime = st.slider(
                "Travel Time to School",
                min_value=1, max_value=4, value=1,
                help="1 - <15 min, 2 - 15-30 min, 3 - 30min-1h, 4 - >1h"
            )
            studytime = st.slider(
                "Weekly Study Time",
                min_value=1, max_value=4, value=2,
                help="1 - <2 hours, 2 - 2-5 hours, 3 - 5-10 hours, 4 - >10 hours"
            )
        
        # Section 3: Academic & Social
        st.markdown("### 📚 Academic & Social Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            failures = st.slider(
                "Number of Past Class Failures",
                min_value=0, max_value=4, value=0
            )
            schoolsup = st.selectbox(
                "Extra Educational Support",
                options=["yes", "no"]
            )
            famsup = st.selectbox(
                "Family Educational Support",
                options=["yes", "no"]
            )
        
        with col2:
            paid = st.selectbox(
                "Extra Paid Classes",
                options=["yes", "no"]
            )
            activities = st.selectbox(
                "Extracurricular Activities",
                options=["yes", "no"]
            )
            nursery = st.selectbox(
                "Attended Nursery School",
                options=["yes", "no"]
            )
        
        with col3:
            higher = st.selectbox(
                "Wants Higher Education",
                options=["yes", "no"]
            )
            internet = st.selectbox(
                "Internet Access at Home",
                options=["yes", "no"]
            )
            romantic = st.selectbox(
                "In Romantic Relationship",
                options=["yes", "no"]
            )
        
        # Section 4: Personal & Health
        st.markdown("### 🏃‍♂️ Personal & Health")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            famrel = st.slider(
                "Family Relationship Quality",
                min_value=1, max_value=5, value=3,
                help="1 - Very bad to 5 - Excellent"
            )
            freetime = st.slider(
                "Free Time After School",
                min_value=1, max_value=5, value=3,
                help="1 - Very low to 5 - Very high"
            )
        
        with col2:
            goout = st.slider(
                "Going Out with Friends",
                min_value=1, max_value=5, value=3,
                help="1 - Very low to 5 - Very high"
            )
            Dalc = st.slider(
                "Workday Alcohol Consumption",
                min_value=1, max_value=5, value=1,
                help="1 - Very low to 5 - Very high"
            )
        
        with col3:
            Walc = st.slider(
                "Weekend Alcohol Consumption",
                min_value=1, max_value=5, value=1,
                help="1 - Very low to 5 - Very high"
            )
            health = st.slider(
                "Current Health Status",
                min_value=1, max_value=5, value=3,
                help="1 - Very bad to 5 - Very good"
            )
        
        # Section 5: Academic Performance
        st.markdown("### 📊 Academic Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            absences = st.slider(
                "Number of School Absences",
                min_value=0, max_value=93, value=5
            )
        
        with col2:
            G1 = st.slider(
                "First Period Grade",
                min_value=0, max_value=20, value=10,
                help="First period grade (0-20)"
            )
        
        with col3:
            G2 = st.slider(
                "Second Period Grade",
                min_value=0, max_value=20, value=10,
                help="Second period grade (0-20)"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Grade", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'school': school, 'sex': sex, 'age': age, 'address': address,
                'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
                'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
                'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
                'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
                'activities': activities, 'nursery': nursery, 'higher': higher,
                'internet': internet, 'romantic': romantic, 'famrel': famrel,
                'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
                'health': health, 'absences': absences, 'G1': G1, 'G2': G2
            }
            
            # Store in session state for use outside form
            st.session_state.last_input_data = input_data
            st.session_state.show_predictions = True
            
            # Make predictions
            with st.spinner("🔄 Making predictions..."):
                try:
                    prediction_utils = st.session_state.prediction_utils
                    
                    # Get predictions from both models
                    rf_pred, rf_class, rf_proba = prediction_utils.predict_random_forest(input_data)
                    xgb_pred, xgb_class, xgb_proba = prediction_utils.predict_xgboost(input_data)
                    
                    # Store predictions in session state
                    st.session_state.last_rf_pred = rf_pred
                    st.session_state.last_rf_class = rf_class
                    st.session_state.last_rf_proba = rf_proba
                    st.session_state.last_xgb_pred = xgb_pred
                    st.session_state.last_xgb_class = xgb_class
                    st.session_state.last_xgb_proba = xgb_proba
                    
                    # Generate recommendations
                    recommendations = prediction_utils.generate_recommendations(input_data, rf_pred)
                    st.session_state.last_recommendations = recommendations
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.session_state.show_predictions = False
    
    # Display predictions outside the form
    if st.session_state.get('show_predictions', False) and 'last_input_data' in st.session_state:
        input_data = st.session_state.last_input_data
        rf_pred = st.session_state.get('last_rf_pred')
        rf_class = st.session_state.get('last_rf_class')
        rf_proba = st.session_state.get('last_rf_proba')
        xgb_pred = st.session_state.get('last_xgb_pred')
        xgb_class = st.session_state.get('last_xgb_class')
        xgb_proba = st.session_state.get('last_xgb_proba')
        recommendations = st.session_state.get('last_recommendations', [])
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌲 Random Forest Prediction")
            
            # Grade prediction with progress bar
            if rf_pred is not None:
                st.metric("Predicted Grade", f"{rf_pred:.2f}/20")
                progress_rf = float(min(rf_pred / 20, 1.0))
            else:
                st.metric("Predicted Grade", "N/A")
                progress_rf = 0.0
            
            if rf_pred is not None and rf_pred >= 16:
                st.success(f"🎉 Excellent performance! ({rf_pred:.2f}/20)")
            elif rf_pred is not None and rf_pred >= 12:
                st.info(f"👍 Good performance ({rf_pred:.2f}/20)")
            elif rf_pred is not None and rf_pred >= 8:
                st.warning(f"⚠️ Average performance ({rf_pred:.2f}/20)")
            elif rf_pred is not None:
                st.error(f"🚨 Needs improvement ({rf_pred:.2f}/20)")
            
            st.progress(progress_rf)
            
            # Classification
            st.write(f"**Grade Category:** {rf_class}")
            if rf_proba is not None:
                max_prob = max(rf_proba.values())
                st.write(f"**Confidence:** {max_prob:.2%}")
        
        with col2:
            st.subheader("🚀 XGBoost Prediction")
            
            # Grade prediction with progress bar
            if xgb_pred is not None:
                st.metric("Predicted Grade", f"{xgb_pred:.2f}/20")
                progress_xgb = float(min(xgb_pred / 20, 1.0))
            else:
                st.metric("Predicted Grade", "N/A")
                progress_xgb = 0.0
            
            if xgb_pred is not None and xgb_pred >= 16:
                st.success(f"🎉 Excellent performance! ({xgb_pred:.2f}/20)")
            elif xgb_pred is not None and xgb_pred >= 12:
                st.info(f"👍 Good performance ({xgb_pred:.2f}/20)")
            elif xgb_pred is not None and xgb_pred >= 8:
                st.warning(f"⚠️ Average performance ({xgb_pred:.2f}/20)")
            elif xgb_pred is not None:
                st.error(f"🚨 Needs improvement ({xgb_pred:.2f}/20)")
            
            st.progress(progress_xgb)
            
            # Classification
            st.write(f"**Grade Category:** {xgb_class}")
            if xgb_proba is not None:
                max_prob = max(xgb_proba.values())
                st.write(f"**Confidence:** {max_prob:.2%}")
        
        # Feature importance and recommendations
        st.subheader("📊 Feature Importance & Explanations")
        
        try:
            # Get SHAP explanations
            prediction_utils = st.session_state.prediction_utils
            shap_values = prediction_utils.get_shap_explanations(input_data)
            
            if shap_values is not None:
                # Display feature importance
                fig = prediction_utils.plot_feature_importance(shap_values, input_data)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.warning(f"⚠️ Could not generate SHAP explanations: {str(e)}")
        
        # Generate recommendations
        st.subheader("💡 Personalized Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Report generation (outside form)
        st.subheader("📄 Generate Report")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Generate CSV Report", use_container_width=True):
                try:
                    report_gen = st.session_state.report_generator
                    csv_data = report_gen.generate_csv_report(
                        input_data, rf_pred, xgb_pred, rf_class, xgb_class, recommendations
                    )
                    
                    st.download_button(
                        label="⬇️ Download CSV Report",
                        data=csv_data,
                        file_name=f"grade_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Error generating CSV report: {str(e)}")
        
        with col2:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                try:
                    report_gen = st.session_state.report_generator
                    pdf_data = report_gen.generate_pdf_report(
                        input_data, rf_pred, xgb_pred, rf_class, xgb_class, recommendations
                    )
                    
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_data,
                        file_name=f"grade_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Error generating PDF report: {str(e)}")

elif page == "📈 Model Performance":
    # Model performance page
    st.header("📈 Model Performance Metrics")
    
    try:
        model_trainer = st.session_state.model_trainer
        metrics = model_trainer.get_model_metrics()
        
        if metrics:
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌲 Random Forest Performance")
                rf_metrics = metrics.get('random_forest', {})
                
                # Regression metrics
                if 'regression' in rf_metrics:
                    reg_metrics = rf_metrics['regression']
                    st.write("**Regression Metrics:**")
                    st.metric("R² Score", f"{reg_metrics.get('r2', 0):.4f}")
                    st.metric("RMSE", f"{reg_metrics.get('rmse', 0):.4f}")
                    st.metric("MAE", f"{reg_metrics.get('mae', 0):.4f}")
                
                # Classification metrics
                if 'classification' in rf_metrics:
                    clf_metrics = rf_metrics['classification']
                    st.write("**Classification Metrics:**")
                    st.metric("Accuracy", f"{clf_metrics.get('accuracy', 0):.4f}")
                    st.metric("F1 Score", f"{clf_metrics.get('f1', 0):.4f}")
                    st.metric("Precision", f"{clf_metrics.get('precision', 0):.4f}")
                    st.metric("Recall", f"{clf_metrics.get('recall', 0):.4f}")
            
            with col2:
                st.subheader("🚀 XGBoost Performance")
                xgb_metrics = metrics.get('xgboost', {})
                
                # Regression metrics
                if 'regression' in xgb_metrics:
                    reg_metrics = xgb_metrics['regression']
                    st.write("**Regression Metrics:**")
                    st.metric("R² Score", f"{reg_metrics.get('r2', 0):.4f}")
                    st.metric("RMSE", f"{reg_metrics.get('rmse', 0):.4f}")
                    st.metric("MAE", f"{reg_metrics.get('mae', 0):.4f}")
                
                # Classification metrics
                if 'classification' in xgb_metrics:
                    clf_metrics = xgb_metrics['classification']
                    st.write("**Classification Metrics:**")
                    st.metric("Accuracy", f"{clf_metrics.get('accuracy', 0):.4f}")
                    st.metric("F1 Score", f"{clf_metrics.get('f1', 0):.4f}")
                    st.metric("Precision", f"{clf_metrics.get('precision', 0):.4f}")
                    st.metric("Recall", f"{clf_metrics.get('recall', 0):.4f}")
        
        else:
            st.info("📊 Model metrics not available. Please train the models first.")
    
    except Exception as e:
        st.error(f"❌ Error loading model metrics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### 🎓 Student Grade Predictor")
st.markdown("*Powered by Machine Learning • Built with Streamlit*")
