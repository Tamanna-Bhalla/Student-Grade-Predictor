import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from .data_processing import DataProcessor

# Try to import SHAP, handle if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try to import ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class PredictionUtils:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            model_files = {
                'rf_regression': 'models/random_forest_regression.joblib',
                'rf_classification': 'models/random_forest_classification.joblib',
                'xgb_regression': 'models/xgboost_regression.joblib',
                'xgb_classification': 'models/xgboost_classification.joblib'
            }
            
            for model_name, filepath in model_files.items():
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
            
            # Load data processor
            self.data_processor.load_encoders()
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_random_forest(self, input_data):
        """Make predictions using Random Forest models"""
        try:
            # Preprocess input
            X = self.data_processor.preprocess_single_input(input_data)
            
            # Regression prediction
            regression_pred = None
            if 'rf_regression' in self.models:
                regression_pred = self.models['rf_regression'].predict(X)[0]
                regression_pred = max(0, min(20, regression_pred))  # Clamp to valid range
            
            # Classification prediction
            classification_pred = None
            classification_proba = None
            if 'rf_classification' in self.models:
                classification_pred = self.models['rf_classification'].predict(X)[0]
                
                # Get probabilities
                proba = self.models['rf_classification'].predict_proba(X)[0]
                classes = self.models['rf_classification'].classes_
                classification_proba = dict(zip(classes, proba))
            
            return regression_pred, classification_pred, classification_proba
            
        except Exception as e:
            print(f"Error in Random Forest prediction: {e}")
            return None, None, None
    
    def predict_xgboost(self, input_data):
        """Make predictions using XGBoost models"""
        try:
            # Preprocess input
            X = self.data_processor.preprocess_single_input(input_data)
            
            # Regression prediction
            regression_pred = None
            if 'xgb_regression' in self.models:
                regression_pred = self.models['xgb_regression'].predict(X)[0]
                regression_pred = max(0, min(20, regression_pred))  # Clamp to valid range
            
            # Classification prediction
            classification_pred = None
            classification_proba = None
            if 'xgb_classification' in self.models:
                classification_pred = self.models['xgb_classification'].predict(X)[0]
                
                # Get probabilities
                proba = self.models['xgb_classification'].predict_proba(X)[0]
                classes = self.models['xgb_classification'].classes_
                classification_proba = dict(zip(classes, proba))
            
            return regression_pred, classification_pred, classification_proba
            
        except Exception as e:
            print(f"Error in XGBoost prediction: {e}")
            return None, None, None
    
    def get_shap_explanations(self, input_data):
        """Get SHAP explanations for predictions"""
        if not SHAP_AVAILABLE:
            return None
            
        try:
            # Use Random Forest regression model for SHAP
            if 'rf_regression' not in self.models:
                return None
                
            model = self.models['rf_regression']
            X = self.data_processor.preprocess_single_input(input_data)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            return shap_values[0]  # Return SHAP values for single prediction
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            return None
    
    def plot_feature_importance(self, shap_values, input_data):
        """Plot feature importance using SHAP values or model feature importance"""
        try:
            if shap_values is not None and self.data_processor.feature_columns:
                # Use SHAP values
                feature_names = self.data_processor.feature_columns
                importance_values = np.abs(shap_values)
                
                # Create DataFrame for plotting
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_values
                }).sort_values('importance', ascending=True)
                
                # Take top 10 features
                top_features = importance_df.tail(10)
                
                fig = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features (SHAP Values)',
                    labels={'importance': 'SHAP Value (Absolute)', 'feature': 'Features'}
                )
                
            else:
                # Fallback to model feature importance
                if 'rf_regression' in self.models:
                    model = self.models['rf_regression']
                    feature_names = self.data_processor.feature_columns
                    importance_values = model.feature_importances_
                    
                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance_values
                    }).sort_values('importance', ascending=True)
                    
                    # Take top 10 features
                    top_features = importance_df.tail(10)
                    
                    fig = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 10 Most Important Features (Model Feature Importance)',
                        labels={'importance': 'Feature Importance', 'feature': 'Features'}
                    )
                else:
                    return None
            
            fig.update_layout(height=500)
            return fig
            
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            return None
    
    def generate_recommendations(self, input_data, predicted_grade):
        """Generate personalized recommendations based on input data and prediction"""
        recommendations = []
        
        try:
            # Study time recommendations
            if input_data.get('studytime', 1) < 3:
                recommendations.append("📚 Increase study time to at least 5-10 hours per week for better academic performance.")
            
            # Attendance recommendations
            if input_data.get('absences', 0) > 10:
                recommendations.append("⏰ Reduce school absences to improve learning consistency and academic outcomes.")
            
            # Family support recommendations
            if input_data.get('famsup', 'no') == 'no':
                recommendations.append("👨‍👩‍👧‍👦 Seek additional family support for educational activities and homework assistance.")
            
            # Higher education aspirations
            if input_data.get('higher', 'yes') == 'no':
                recommendations.append("🎓 Consider the benefits of higher education for long-term career prospects.")
            
            # Extracurricular activities
            if input_data.get('activities', 'no') == 'no':
                recommendations.append("🏃‍♂️ Participate in extracurricular activities to develop well-rounded skills.")
            
            # Internet access
            if input_data.get('internet', 'yes') == 'no':
                recommendations.append("🌐 Improve internet access for better educational resources and research opportunities.")
            
            # Health recommendations
            if input_data.get('health', 3) < 3:
                recommendations.append("🏥 Focus on improving physical health through proper diet, exercise, and medical care.")
            
            # Social life balance
            if input_data.get('goout', 3) > 4:
                recommendations.append("⚖️ Balance social activities with academic responsibilities for optimal performance.")
            
            # Alcohol consumption
            if input_data.get('Dalc', 1) > 2 or input_data.get('Walc', 1) > 2:
                recommendations.append("🚫 Reduce alcohol consumption to improve focus and academic performance.")
            
            # Failures and additional support
            if input_data.get('failures', 0) > 0:
                recommendations.append("📖 Consider additional tutoring or educational support to address past academic challenges.")
            
            # Grade-specific recommendations
            if predicted_grade < 10:
                recommendations.append("🆘 Seek immediate academic intervention and counseling support.")
                recommendations.append("👨‍🏫 Work closely with teachers to identify specific areas for improvement.")
            elif predicted_grade < 15:
                recommendations.append("📈 Focus on consistent study habits and seek help in challenging subjects.")
            
            # If no specific recommendations, provide general advice
            if not recommendations:
                recommendations.append("✨ Continue your excellent work! Maintain current study habits and stay motivated.")
                recommendations.append("🎯 Set specific academic goals and track your progress regularly.")
            
            return recommendations[:8]  # Limit to top 8 recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["📚 Focus on consistent study habits and seek support when needed."]

class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_csv_report(self, input_data, rf_pred, xgb_pred, rf_class, xgb_class, recommendations):
        """Generate CSV report"""
        try:
            # Prepare report data
            report_data = {
                'Report Generated': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Random Forest Prediction': [f"{rf_pred:.2f}" if rf_pred else "N/A"],
                'XGBoost Prediction': [f"{xgb_pred:.2f}" if xgb_pred else "N/A"],
                'RF Grade Category': [rf_class if rf_class else "N/A"],
                'XGB Grade Category': [xgb_class if xgb_class else "N/A"],
                'Average Prediction': [f"{(rf_pred + xgb_pred) / 2:.2f}" if rf_pred and xgb_pred else "N/A"]
            }
            
            # Add input data
            for key, value in input_data.items():
                report_data[f'Input_{key}'] = [value]
            
            # Add recommendations
            for i, rec in enumerate(recommendations[:5], 1):
                report_data[f'Recommendation_{i}'] = [rec]
            
            # Create DataFrame and convert to CSV
            df = pd.DataFrame(report_data)
            
            # Convert to CSV string
            output = StringIO()
            df.to_csv(output, index=False)
            csv_string = output.getvalue()
            output.close()
            
            return csv_string
            
        except Exception as e:
            print(f"Error generating CSV report: {e}")
            return "Error generating report"
    
    def generate_pdf_report(self, input_data, rf_pred, xgb_pred, rf_class, xgb_class, recommendations):
        """Generate PDF report"""
        if not REPORTLAB_AVAILABLE:
            return "PDF generation not available. ReportLab library not installed."
        
        try:
            # Create PDF buffer
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading2']
            normal_style = styles['Normal']
            
            # Title
            story.append(Paragraph("Student Grade Prediction Report", title_style))
            story.append(Spacer(1, 12))
            
            # Report info
            story.append(Paragraph(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Spacer(1, 12))
            
            # Predictions
            story.append(Paragraph("Prediction Results", heading_style))
            if rf_pred and xgb_pred:
                predictions_data = [
                    ['Model', 'Predicted Grade', 'Grade Category'],
                    ['Random Forest', f"{rf_pred:.2f}/20", rf_class],
                    ['XGBoost', f"{xgb_pred:.2f}/20", xgb_class],
                    ['Average', f"{(rf_pred + xgb_pred) / 2:.2f}/20", ""]
                ]
                
                predictions_table = Table(predictions_data)
                predictions_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(predictions_table)
            story.append(Spacer(1, 12))
            
            # Student Information
            story.append(Paragraph("Student Information", heading_style))
            student_data = []
            for key, value in input_data.items():
                student_data.append([key.replace('_', ' ').title(), str(value)])
            
            # Split into chunks for better formatting
            chunk_size = 15
            for i in range(0, len(student_data), chunk_size):
                chunk = student_data[i:i + chunk_size]
                
                student_table = Table(chunk)
                student_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                
                story.append(student_table)
                story.append(Spacer(1, 6))
            
            # Recommendations
            story.append(Spacer(1, 12))
            story.append(Paragraph("Personalized Recommendations", heading_style))
            
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", normal_style))
                story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            
            # Get PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return f"Error generating PDF report: {e}"
