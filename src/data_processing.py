import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path='data/student_data.csv'):
        """Load the student dataset"""
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                return df
            else:
                # Create synthetic data if file doesn't exist
                return self._create_synthetic_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic dataset matching the Kaggle structure"""
        np.random.seed(42)
        n_samples = 395  # Similar to original dataset size
        
        # Create synthetic data matching the real dataset structure
        data = {
            'school': np.random.choice(['GP', 'MS'], n_samples, p=[0.88, 0.12]),
            'sex': np.random.choice(['F', 'M'], n_samples, p=[0.53, 0.47]),
            'age': np.random.choice(range(15, 23), n_samples),
            'address': np.random.choice(['U', 'R'], n_samples, p=[0.78, 0.22]),
            'famsize': np.random.choice(['LE3', 'GT3'], n_samples, p=[0.29, 0.71]),
            'Pstatus': np.random.choice(['T', 'A'], n_samples, p=[0.90, 0.10]),
            'Medu': np.random.choice(range(5), n_samples),
            'Fedu': np.random.choice(range(5), n_samples),
            'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
            'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
            'reason': np.random.choice(['home', 'reputation', 'course', 'other'], n_samples),
            'guardian': np.random.choice(['mother', 'father', 'other'], n_samples),
            'traveltime': np.random.choice(range(1, 5), n_samples),
            'studytime': np.random.choice(range(1, 5), n_samples),
            'failures': np.random.choice(range(4), n_samples, p=[0.7, 0.2, 0.07, 0.03]),
            'schoolsup': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85]),
            'famsup': np.random.choice(['yes', 'no'], n_samples, p=[0.56, 0.44]),
            'paid': np.random.choice(['yes', 'no'], n_samples, p=[0.10, 0.90]),
            'activities': np.random.choice(['yes', 'no'], n_samples, p=[0.51, 0.49]),
            'nursery': np.random.choice(['yes', 'no'], n_samples, p=[0.82, 0.18]),
            'higher': np.random.choice(['yes', 'no'], n_samples, p=[0.92, 0.08]),
            'internet': np.random.choice(['yes', 'no'], n_samples, p=[0.83, 0.17]),
            'romantic': np.random.choice(['yes', 'no'], n_samples, p=[0.37, 0.63]),
            'famrel': np.random.choice(range(1, 6), n_samples),
            'freetime': np.random.choice(range(1, 6), n_samples),
            'goout': np.random.choice(range(1, 6), n_samples),
            'Dalc': np.random.choice(range(1, 6), n_samples, p=[0.7, 0.15, 0.1, 0.03, 0.02]),
            'Walc': np.random.choice(range(1, 6), n_samples, p=[0.6, 0.2, 0.12, 0.05, 0.03]),
            'health': np.random.choice(range(1, 6), n_samples),
            'absences': np.random.poisson(5, n_samples),
            'G1': np.random.choice(range(21), n_samples),
            'G2': np.random.choice(range(21), n_samples),
        }
        
        # Create target variable G3 with some correlation to other grades
        G3 = []
        for i in range(n_samples):
            base_grade = (data['G1'][i] + data['G2'][i]) / 2
            # Add some noise and correlation with other factors
            adjustment = 0
            if data['studytime'][i] >= 3:
                adjustment += 1
            if data['failures'][i] == 0:
                adjustment += 1
            if data['higher'][i] == 'yes':
                adjustment += 0.5
            if data['absences'][i] > 10:
                adjustment -= 1
            
            final_grade = base_grade + adjustment + np.random.normal(0, 2)
            G3.append(max(0, min(20, int(final_grade))))
        
        data['G3'] = G3
        
        df = pd.DataFrame(data)
        
        # Save synthetic data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/student_data.csv', index=False)
        
        return df
    
    def preprocess_data(self, df, fit_encoders=True):
        """Preprocess the data for machine learning"""
        df_processed = df.copy()
        
        # Identify categorical and numerical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target variable from features
        if 'G3' in numerical_columns:
            numerical_columns.remove('G3')
        
        # Encode categorical variables
        for col in categorical_columns:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen labels
                    le = self.label_encoders[col]
                    # Get unique values in current data
                    unique_vals = df_processed[col].unique()
                    # For unseen values, assign them to the most frequent class
                    most_frequent_class = 0  # Default to first class
                    
                    encoded_vals = []
                    for val in df_processed[col]:
                        if val in le.classes_:
                            encoded_vals.append(le.transform([val])[0])
                        else:
                            encoded_vals.append(most_frequent_class)
                    df_processed[col] = encoded_vals
        
        # Store feature columns for later use
        if fit_encoders:
            self.feature_columns = categorical_columns + numerical_columns
        
        # Return features and target
        X = df_processed[self.feature_columns]
        y = df_processed['G3'] if 'G3' in df_processed.columns else None
        
        return X, y
    
    def preprocess_single_input(self, input_data):
        """Preprocess a single input for prediction"""
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocess without fitting encoders
        X, _ = self.preprocess_data(df, fit_encoders=False)
        
        return X
    
    def create_grade_categories(self, grades):
        """Create grade categories from numerical grades"""
        categories = []
        for grade in grades:
            if grade >= 16:
                categories.append('Excellent')
            elif grade >= 12:
                categories.append('Good')
            elif grade >= 8:
                categories.append('Average')
            else:
                categories.append('Poor')
        return categories
    
    def save_encoders(self, path='models/'):
        """Save label encoders and scaler"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.label_encoders, os.path.join(path, 'label_encoders.joblib'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        joblib.dump(self.feature_columns, os.path.join(path, 'feature_columns.joblib'))
    
    def load_encoders(self, path='models/'):
        """Load label encoders and scaler"""
        try:
            self.label_encoders = joblib.load(os.path.join(path, 'label_encoders.joblib'))
            self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
            self.feature_columns = joblib.load(os.path.join(path, 'feature_columns.joblib'))
            return True
        except:
            return False
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
