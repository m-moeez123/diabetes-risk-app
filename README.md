# AI-Driven Diabetes Risk Profiling and Stratification

## Project Overview

This project demonstrates an AI-driven approach for precision risk profiling and stratification for diabetes and related complications using the Diabetes Health Indicators Dataset. The goal is to predict an individual's risk of having diabetes or prediabetes, stratify individuals into complication risk tiers, and provide personalized recommendations based on the factors influencing their predicted risk.

The project utilizes machine learning models, risk stratification techniques, and explainability methods (SHAP) to provide a more nuanced and personalized understanding of diabetes risk. A Streamlit web application is developed to showcase the project's capabilities, allowing users to input their health data and receive their risk assessment and personalized insights.

## Dataset

The project uses the Diabetes Health Indicators Dataset, based on the 2015 Behavioral Risk Factor Surveillance System (BRFSS). The dataset contains various health-related indicators and the diabetes status of respondents.

Key features include:
*   **Binary Indicators:** `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.
*   **Ordinal Indicators:** `GenHlth`, `Education`, `Income`, `Age`.
*   **Continuous Indicators:** `BMI`, `MentHlth`, `PhysHlth`.

The target variable is a binary indicator of whether an individual has diabetes or prediabetes (`Diabetes_binary`).

## Methodology

The project follows a multi-stage methodology:

1.  **Data Loading and Preprocessing:** The dataset is loaded, and a binary target variable is created. Features are identified as binary, ordinal, or continuous. Continuous features are scaled using `StandardScaler`.
2.  **Data Splitting:** The data is split into training, validation, and testing sets using stratified sampling to maintain the class distribution.
3.  **Model Selection and Training:** Several machine learning models (Logistic Regression, Random Forest, LightGBM, XGBoost) and a Deep Learning model are trained for binary diabetes risk prediction, addressing class imbalance.
4.  **Model Evaluation:** Models are evaluated on the validation set using metrics like Accuracy, Precision, Recall, F1-score, ROC AUC, and Brier Score Loss.
5.  **Risk Stratification (Complication Risk):** A `Complication_Proxy_Score` is created based on key health indicators. Individuals are stratified into Low, Moderate, and High complication risk tiers based on this score.
6.  **Risk Group Analysis:** The characteristics and true diabetes prevalence within each complication risk tier are analyzed.
7.  **Explainability (SHAP):** SHAP values are calculated for the trained XGBoost model to understand global feature importance and individual prediction drivers.
8.  **Personalized Recommendations:** Personalized recommendations are generated based on the top features influencing an individual's predicted risk (identified by SHAP).
9.  **Web Application Development:** A Streamlit application is built to allow users to interact with the trained model and receive personalized risk assessments and recommendations.

## Key Findings

*   **Diabetes Risk Prediction:** The trained models, particularly XGBoost and the Deep Learning model, demonstrated promising performance in predicting diabetes/prediabetes risk, even with the class imbalance.
*   **Important Risk Factors:** SHAP analysis highlighted key factors influencing diabetes risk, including demographic features and various health indicators like BMI, HighBP, and smoking status.
*   **Complication Risk Stratification:** The `Complication_Proxy_Score` and derived risk tiers effectively stratify individuals based on their potential for diabetes-related complications, showing a clear correlation with diabetes prevalence.
*   **Personalized Insights:** SHAP explainability enables the identification of individual risk drivers, allowing for tailored and actionable health recommendations.

## Streamlit Web Application

A Streamlit application is provided to demonstrate the project. It allows users to:

*   Input their health data.
*   Get their predicted diabetes probability.
*   Identify the top factors influencing their prediction (SHAP).
*   Determine their complication risk tier.
*   Receive personalized recommendations.

### Running the Streamlit Application Locally

1.  **Clone the repository:**

2.      pip install -r requirements.txt

3.      streamlit run app.py

4.  **APP LINK**
5.  https://diabetes-risk-app-g5dmggwdhqpjtnxkpdshwd.streamlit.app/
