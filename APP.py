import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Load the saved artifacts
# Define the directory where artifacts were saved
artifact_dir = "diabetes_app_artifacts"  # Use relative path
model_path = os.path.join(artifact_dir, 'xgb_model.joblib')
preprocessor_path = os.path.join(artifact_dir, 'preprocessor.joblib')
# Load the fitted ColumnTransformer
preprocessor_path = os.path.join(artifact_dir, 'preprocessor.joblib')
preprocessor = joblib.load(preprocessor_path)

# Load the fitted StandardScaler used for clustering (if needed)
# scaler_path = os.path.join(artifact_dir, 'clustering_scaler.joblib')
# scaler = joblib.load(scaler_path)

# Define risk thresholds (copying from previous analysis)
low_risk_threshold_cps = 3.20
moderate_risk_threshold_cps = 4.50

# Define personalized recommendations (copying from previous analysis)
personalized_recommendations = {
    'Sex': 'Consider targeted health screening based on demographic risk.',
    'BMI': 'Focus on weight management through diet and exercise.',
    'HighBP': 'Work with a healthcare provider to manage blood pressure.',
    'Income': 'Explore resources for affordable healthy food and healthcare access.',
    'Smoker': 'Seek support for smoking cessation.',
    'Education': 'Increase awareness of diabetes risk factors and healthy lifestyle choices.',
    'DiffWalk': 'Incorporate regular, gentle physical activity tailored to mobility levels.',
    'Stroke': 'Focus on managing cardiovascular health to prevent further complications.',
    'Fruits': 'Increase daily fruit intake as part of a balanced diet.',
    'HighChol': 'Work with a healthcare provider to manage cholesterol levels.'
}

# Define column names (ensure these match the columns used in training)
feature_names = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
               'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
               'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'] # Based on X.columns after binary target creation


# Set the title of the Streamlit application
st.title('AI-Driven Diabetes Risk Profiling and Stratification')

st.write("Input your health indicators to get your predicted diabetes risk, top influencing factors, complication risk tier, and personalized recommendations.")

# Create input sections for each feature
st.header("User Health Indicators")

# Define feature types and ranges based on the original dataset description and previous analysis
# Referencing feature_names

# Binary Features (0 or 1) - using selectbox for clarity
binary_inputs = {}
binary_cols = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
               'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
               'NoDocbcCost', 'DiffWalk', 'Sex']

st.subheader("Binary Indicators")
for col in binary_cols:
    if col == 'Sex':
        binary_inputs[col] = st.selectbox(
            f"{col} (0: Female, 1: Male)", 
            [0.0, 1.0], 
            format_func=lambda x: 'Female' if x == 0.0 else 'Male',
            key=f"{col}_selectbox"
        )
    elif col == 'CholCheck':
        binary_inputs[col] = st.selectbox(
            f"{col} (0: No, 1: Yes)", 
            [0.0, 1.0], 
            format_func=lambda x: 'No' if x == 0.0 else 'Yes',
            key=f"{col}_selectbox"
        )
    elif col in ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'DiffWalk', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost']:
        binary_inputs[col] = st.selectbox(
            f"{col} (0: No, 1: Yes)", 
            [0.0, 1.0], 
            format_func=lambda x: 'No' if x == 0.0 else 'Yes',
            key=f"{col}_selectbox"
        )


# Ordinal Features (Integer values with order) - using number_input or selectbox
ordinal_inputs = {}
ordinal_cols = ['GenHlth', 'Education', 'Income', 'Age']

st.subheader("Ordinal Indicators")
# Assuming the ranges from the dataset description (see describe() output)
ordinal_inputs['GenHlth'] = st.selectbox("General Health (1: Excellent, 5: Poor)", list(range(1, 6)), format_func=lambda x: ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][x-1])
# Education: 1=Never attended school or only kindergarten, 2=Grades 1-8, 3=Grades 9-11, 4=High school graduate, 5=Some college or technical school, 6=College graduate
ordinal_inputs['Education'] = st.selectbox("Education Level", list(range(1, 7)), format_func=lambda x: ['Never attended school/kindergarten', 'Grades 1-8', 'Grades 9-11', 'High school graduate', 'Some college/technical school', 'College graduate'][x-1])
# Income: 1=Less than $10k, 2=$10-15k, 3=$15-20k, 4=$20-25k, 5=$25-35k, 6=$35-50k, 7=$50-75k, 8=$75k or more
ordinal_inputs['Income'] = st.selectbox("Income Level", list(range(1, 9)), format_func=lambda x: ['< $10k', '$10-15k', '$15-20k', '$20-25k', '$25-35k', '$35-50k', '$50-75k', '$75k+'][x-1])
# Age: 1=18-24, 2=25-29, ..., 13=80+
ordinal_inputs['Age'] = st.selectbox("Age Group", list(range(1, 14)), format_func=lambda x: f"{17 + x*5}-{23 + x*5}" if x < 13 else "80+")


# Continuous Features (Numerical values) - using number_input
continuous_inputs = {}
continuous_cols = ['BMI', 'MentHlth', 'PhysHlth']

st.subheader("Continuous Indicators")
# Using typical ranges observed in the data
continuous_inputs['BMI'] = st.number_input("BMI", min_value=12.0, max_value=98.0, value=25.0, step=0.1)
continuous_inputs['MentHlth'] = st.number_input("Days of poor mental health in past 30 days", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
continuous_inputs['PhysHlth'] = st.number_input("Days of poor physical health in past 30 days", min_value=0.0, max_value=30.0, value=0.0, step=1.0)


# Button to trigger prediction
st.header("Predict and Analyze")
predict_button = st.button('Predict Risk and Get Recommendations')

# Create placeholders for output sections
st.header("Results")

# Placeholder for Predicted Probability
st.subheader("Predicted Diabetes Probability")
probability_placeholder = st.empty()

# Placeholder for Top Influencing Factors (SHAP)
st.subheader("Top Influencing Factors")
shap_placeholder = st.empty()

# Placeholder for Complication Risk Tier
st.subheader("Complication Risk Tier")
risk_tier_placeholder = st.empty()

# Placeholder for Personalized Recommendations
st.subheader("Personalized Recommendations")
recommendations_placeholder = st.empty()

# Placeholder for Phenotype Cluster (Optional)
st.subheader("Patient Phenotype Cluster")

cluster_placeholder = st.empty()

if predict_button:
    # Combine all inputs into a single dictionary
    user_input = {}
    user_input.update(binary_inputs)
    user_input.update(ordinal_inputs)
    user_input.update(continuous_inputs)

    # Create a DataFrame for the model
    user_input_df = pd.DataFrame([user_input], columns=feature_names)

    # Preprocess the input
    processed_input = preprocessor.transform(user_input_df)

    # Load the model (if not already loaded above)
    xgb_model = joblib.load(model_path)

    # Predict probability
    probability = xgb_model.predict_proba(processed_input)[0][1]
    probability_placeholder.write(f"{probability:.4f}")

    # SHAP values for explanation
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(processed_input)
    top_features_idx = np.argsort(np.abs(shap_values[0]))[::-1][:5]
    top_features = [feature_names[i] for i in top_features_idx]
    shap_placeholder.write(f"Top factors: {', '.join(top_features)}")

    # Complication risk tier
    if probability < low_risk_threshold_cps:
        risk_tier = "Low"
    elif probability < moderate_risk_threshold_cps:
        risk_tier = "Moderate"
    else:
        risk_tier = "High"
    risk_tier_placeholder.write(f"Risk Tier: {risk_tier}")

    # Personalized recommendations
    recommendations_list = [personalized_recommendations.get(f, "") for f in top_features if f in personalized_recommendations]
    recommendations_placeholder.write("\n".join(recommendations_list))

    # (Optional) Phenotype cluster logic can go here

