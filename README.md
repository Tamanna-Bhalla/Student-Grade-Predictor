# Student Grade Predictor

> An explainable machine learning application for predicting student performance and turning model output into practical, personalized improvement guidance.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-modeling-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-boosting-189FDD)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2)](https://shap.readthedocs.io/)

## Overview

Student Grade Predictor is an end-to-end ML project that estimates a student's final grade from demographic, academic, social, and behavioral factors. It combines regression and classification models with an interactive Streamlit interface, model explainability, and automatically generated student reports.

The project is designed to support **early intervention**: instead of only returning a score, it highlights influential factors and translates the prediction into actionable recommendations.

## Highlights

- Predicts final grade (`G3`) on a 0-20 scale.
- Compares **Random Forest** and **XGBoost** models.
- Supports both:
  - Regression: predicted numeric grade.
  - Classification: `Poor`, `Average`, `Good`, or `Excellent`.
- Explains predictions with SHAP feature contributions.
- Generates personalized improvement recommendations based on student inputs.
- Produces downloadable PDF and CSV reports.
- Includes interactive data exploration, grade distributions, correlations, and model performance views.
- Loads pre-trained model artifacts or trains them automatically when needed.

## Application workflow

```text
Student profile
      |
      v
Data preprocessing and categorical encoding
      |
      +--> Random Forest regression/classification
      |
      +--> XGBoost regression/classification
      |
      v
Grade prediction + performance category
      |
      +--> SHAP explanation
      +--> Personalized recommendations
      +--> Downloadable PDF/CSV report
```

## Tech stack

| Area | Technologies |
| --- | --- |
| Interface | Streamlit |
| Data processing | Python, Pandas, NumPy |
| Machine learning | scikit-learn, Random Forest, XGBoost |
| Explainability | SHAP |
| Visualizations | Plotly, Seaborn |
| Reporting | ReportLab, CSV |
| Model persistence | Joblib |

## Project structure

```text
Student-Grade-Predictor/
├── data/
│   └── student_data.csv
├── models/
│   ├── random_forest_classification.joblib
│   ├── random_forest_regression.joblib
│   ├── xgboost_classification.joblib
│   ├── xgboost_regression.joblib
│   ├── label_encoders.joblib
│   └── scaler.joblib
├── reports/
├── src/
│   ├── data_processing.py
│   ├── model.py
│   └── utils.py
├── streamlit_app.py
├── requirements.txt
└── run_no_venv.ps1
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Tamanna-Bhalla/Student-Grade-Predictor.git
cd Student-Grade-Predictor
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run streamlit_app.py
```

The app opens in your browser. Use the sidebar to explore the data, enter a student profile, review model predictions, inspect feature explanations, and download a report.

### Windows shortcut

If you prefer not to create a virtual environment, the included helper script installs dependencies into your user site-packages:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_no_venv.ps1
```

## Model details

The application uses the UCI Student Performance dataset structure with 395 records and 32 input features. Categorical variables are label-encoded and model artifacts are persisted with Joblib for repeatable predictions.

The target variable is `G3`, the student's final grade. Classification labels are derived from the predicted grade:

| Grade range | Category |
| --- | --- |
| 0-7 | Poor |
| 8-11 | Average |
| 12-15 | Good |
| 16-20 | Excellent |

SHAP values are calculated for the Random Forest regression model to show which factors most influenced an individual prediction. Recommendations are generated from study time, absences, previous failures, support systems, health, and related student inputs.

## Responsible use

This project is intended for educational demonstration and exploratory analysis. Predictions should support, not replace, teacher judgment, student context, or formal academic intervention. The dataset is relatively small, and model outputs may reflect historical bias or incomplete information.

## Resume-ready summary

**Student Grade Predictor** - Built an explainable student-performance prediction system using Random Forest and XGBoost, served through an interactive Streamlit application. Implemented dual regression/classification workflows, SHAP-based feature explanations, personalized recommendations, and automated PDF/CSV reporting.

## License

This project is available for personal, educational, and portfolio use. Add a license file before distributing it as an open-source project.
