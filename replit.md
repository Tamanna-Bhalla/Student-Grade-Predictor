# Overview

Student Grade Predictor is a full-featured web application built with Streamlit that uses machine learning to predict student academic performance. The application collects comprehensive student profile data including academic history, personal circumstances, and environmental factors to predict both numerical scores and grade categories. It features interactive visualizations, model explainability through feature importance analysis, and generates actionable recommendations for academic improvement.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses **Streamlit** as the primary web framework, providing a modern, responsive interface with:
- Interactive form components for data collection with validation and tooltips
- Real-time prediction display with progress bars and color-coded status indicators
- Interactive Plotly visualizations for data exploration and feature importance
- Downloadable reports in multiple formats (PDF/CSV)
- Session state management for maintaining user data and model instances

## Backend Architecture
The codebase follows a **modular architecture** with clear separation of concerns:

### Core Components
- **DataProcessor** (`src/data_processing.py`): Handles data loading, preprocessing, synthetic data generation, and feature engineering with label encoding and standardization
- **ModelTrainer** (`src/model.py`): Manages training of multiple ML models including Random Forest and XGBoost for both regression and classification tasks
- **PredictionUtils** (`src/utils.py`): Provides prediction functionality, model loading, and feature importance analysis
- **ReportGenerator** (`src/utils.py`): Handles PDF and CSV report generation with recommendations

### Model Strategy
The application implements a **dual-model approach**:
- **Regression models** for predicting exact percentage scores
- **Classification models** for predicting grade categories (Excellent/Good/Average/Poor)
- Support for both Random Forest and XGBoost algorithms
- Model persistence using joblib for efficient loading

### Data Processing Pipeline
- **Synthetic data generation** when real datasets are unavailable
- **Feature preprocessing** with label encoding for categorical variables and standardization for numerical features
- **Grade categorization** system for classification tasks
- **Input validation** and error handling throughout the pipeline

## Machine Learning Components
- **Feature Engineering**: Comprehensive preprocessing of 19+ student attributes including academic, personal, and environmental factors
- **Model Explainability**: Integration with SHAP (when available) for feature importance analysis
- **Performance Metrics**: Comprehensive evaluation using R², RMSE, MAE for regression and accuracy, precision, recall, F1-score for classification
- **Recommendation Engine**: Rule-based system that generates actionable advice based on prediction results and feature importance

## User Interface Design
- **Progressive disclosure** with organized form sections (Profile, Academics, Environment)
- **Visual feedback** through progress bars, icons, and color-coded status messages
- **Interactive charts** for exploring predictions and feature importance
- **Responsive design** optimized for various screen sizes

# External Dependencies

## Core ML and Data Processing
- **scikit-learn**: Primary machine learning framework for Random Forest models, preprocessing, and evaluation metrics
- **XGBoost**: Advanced gradient boosting framework for improved prediction accuracy
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **joblib**: Model serialization and persistence

## Web Framework and Visualization
- **Streamlit**: Web application framework and UI components
- **Plotly**: Interactive data visualization and charting (plotly.express, plotly.graph_objects)

## Optional Enhanced Features
- **SHAP**: Model explainability and feature importance analysis (graceful degradation if unavailable)
- **ReportLab**: PDF report generation with professional formatting (fallback to CSV if unavailable)

## Development and Deployment
- **warnings**: Error suppression for cleaner user experience
- **os/sys**: File system operations and path management
- **datetime**: Timestamp handling for reports
- **io**: In-memory file operations for report generation

The application is designed with **graceful degradation** - core functionality remains available even if optional dependencies like SHAP or ReportLab are missing, ensuring broad compatibility across different deployment environments.