import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from .data_processing import DataProcessor

class ModelTrainer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = {}
        self.metrics = {}
        
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest models for both regression and classification"""
        
        # Regression model
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_reg.fit(X_train, y_train)
        
        # Predictions
        y_pred_reg = rf_reg.predict(X_test)
        
        # Regression metrics
        reg_metrics = {
            'r2': r2_score(y_test, y_pred_reg),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_reg)),
            'mae': mean_absolute_error(y_test, y_pred_reg)
        }
        
        # Classification model
        y_train_cat = self.data_processor.create_grade_categories(y_train)
        y_test_cat = self.data_processor.create_grade_categories(y_test)
        
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_clf.fit(X_train, y_train_cat)
        
        # Predictions
        y_pred_clf = rf_clf.predict(X_test)
        
        # Classification metrics
        clf_metrics = {
            'accuracy': accuracy_score(y_test_cat, y_pred_clf),
            'precision': precision_score(y_test_cat, y_pred_clf, average='weighted', zero_division='warn'),
            'recall': recall_score(y_test_cat, y_pred_clf, average='weighted', zero_division='warn'),
            'f1': f1_score(y_test_cat, y_pred_clf, average='weighted', zero_division='warn')
        }
        
        # Store models and metrics
        self.models['rf_regression'] = rf_reg
        self.models['rf_classification'] = rf_clf
        self.metrics['random_forest'] = {
            'regression': reg_metrics,
            'classification': clf_metrics
        }
        
        return rf_reg, rf_clf, reg_metrics, clf_metrics
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost models for both regression and classification"""
        
        # Regression model
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_reg.fit(X_train, y_train)
        
        # Predictions
        y_pred_reg = xgb_reg.predict(X_test)
        
        # Regression metrics
        reg_metrics = {
            'r2': r2_score(y_test, y_pred_reg),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_reg)),
            'mae': mean_absolute_error(y_test, y_pred_reg)
        }
        
        # Classification model
        y_train_cat = self.data_processor.create_grade_categories(y_train)
        y_test_cat = self.data_processor.create_grade_categories(y_test)
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_clf.fit(X_train, y_train_cat)
        
        # Predictions
        y_pred_clf = xgb_clf.predict(X_test)
        
        # Classification metrics
        clf_metrics = {
            'accuracy': accuracy_score(y_test_cat, y_pred_clf),
            'precision': precision_score(y_test_cat, y_pred_clf, average='weighted', zero_division='warn'),
            'recall': recall_score(y_test_cat, y_pred_clf, average='weighted', zero_division='warn'),
            'f1': f1_score(y_test_cat, y_pred_clf, average='weighted', zero_division='warn')
        }
        
        # Store models and metrics
        self.models['xgb_regression'] = xgb_reg
        self.models['xgb_classification'] = xgb_clf
        self.metrics['xgboost'] = {
            'regression': reg_metrics,
            'classification': clf_metrics
        }
        
        return xgb_reg, xgb_clf, reg_metrics, clf_metrics
    
    def train_all_models(self):
        """Train all models"""
        # Load and preprocess data
        df = self.data_processor.load_data()
        X, y = self.data_processor.preprocess_data(df, fit_encoders=True)
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
        
        # Train models
        print("Training Random Forest models...")
        self.train_random_forest(X_train, X_test, y_train, y_test)
        
        print("Training XGBoost models...")
        self.train_xgboost(X_train, X_test, y_train, y_test)
        
        # Save models and preprocessors
        self.save_models()
        self.data_processor.save_encoders()
        
        print("All models trained and saved successfully!")
        
        return self.metrics
    
    def save_models(self, path='models/'):
        """Save all trained models"""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        if 'rf_regression' in self.models:
            joblib.dump(self.models['rf_regression'], os.path.join(path, 'random_forest_regression.joblib'))
        
        if 'rf_classification' in self.models:
            joblib.dump(self.models['rf_classification'], os.path.join(path, 'random_forest_classification.joblib'))
        
        if 'xgb_regression' in self.models:
            joblib.dump(self.models['xgb_regression'], os.path.join(path, 'xgboost_regression.joblib'))
        
        if 'xgb_classification' in self.models:
            joblib.dump(self.models['xgb_classification'], os.path.join(path, 'xgboost_classification.joblib'))
        
        # Save metrics
        if self.metrics:
            joblib.dump(self.metrics, os.path.join(path, 'model_metrics.joblib'))
    
    def load_models(self, path='models/'):
        """Load all trained models"""
        try:
            # Load models
            model_files = {
                'rf_regression': 'random_forest_regression.joblib',
                'rf_classification': 'random_forest_classification.joblib',
                'xgb_regression': 'xgboost_regression.joblib',
                'xgb_classification': 'xgboost_classification.joblib'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(path, filename)
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
            
            # Load metrics
            metrics_path = os.path.join(path, 'model_metrics.joblib')
            if os.path.exists(metrics_path):
                self.metrics = joblib.load(metrics_path)
            
            # Load data processor
            self.data_processor.load_encoders(path)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_model_metrics(self):
        """Get model performance metrics"""
        if not self.metrics:
            # Try to load metrics
            try:
                metrics_path = 'models/model_metrics.joblib'
                if os.path.exists(metrics_path):
                    self.metrics = joblib.load(metrics_path)
            except:
                pass
        
        return self.metrics
    
    def print_model_performance(self):
        """Print model performance metrics"""
        if not self.metrics:
            print("No metrics available. Please train models first.")
            return
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        
        for model_name, model_metrics in self.metrics.items():
            print(f"\n{model_name.upper()} PERFORMANCE:")
            print("-" * 30)
            
            if 'regression' in model_metrics:
                reg_metrics = model_metrics['regression']
                print("Regression Metrics:")
                print(f"  R² Score: {reg_metrics['r2']:.4f}")
                print(f"  RMSE: {reg_metrics['rmse']:.4f}")
                print(f"  MAE: {reg_metrics['mae']:.4f}")
            
            if 'classification' in model_metrics:
                clf_metrics = model_metrics['classification']
                print("Classification Metrics:")
                print(f"  Accuracy: {clf_metrics['accuracy']:.4f}")
                print(f"  Precision: {clf_metrics['precision']:.4f}")
                print(f"  Recall: {clf_metrics['recall']:.4f}")
                print(f"  F1-Score: {clf_metrics['f1']:.4f}")
