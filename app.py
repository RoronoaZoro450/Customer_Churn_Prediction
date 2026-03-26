import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ================== HELPERS ==================

def bin_mapping(X):
    X = X.copy()
    binary_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    for col in binary_cols:
        X[col] = X[col].map({
            'Yes': 1,
            'No': 0,
            'No internet service': 0
        })
    return X


def clean_feature_name(feature):
    if "cat__" in feature:
        parts = feature.split("__")[1]
        if "_" in parts:
            col, val = parts.split("_", 1)
            return f"{col} = {val}"
    elif "num__" in feature:
        return feature.split("__")[1]
    elif "bin__" in feature:
        return feature.split("__")[1]
    return feature


def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "bin":
            feature_names.extend(cols)

        elif name == "num":
            feature_names.extend(cols)

        elif name == "cat":
            ohe = transformer
            encoded = ohe.get_feature_names_out(cols)
            feature_names.extend(encoded)

    return feature_names


def load_model():
    return joblib.load("churn_pipeline_v2.pkl")


# ================== CORE FUNCTION ==================

def predict_churn_and_shap_analysis(input_df):
    model = load_model()

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    X_transformed = preprocessor.transform(input_df)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_transformed)

    return prediction, probability, shap_values, preprocessor


# ================== STREAMLIT UI ==================

st.title("📊 Customer Churn Prediction System")

option = ["Yes", "No", "No internet service"]

with st.sidebar:
    st.header("Enter Customer Details")

    tenure = st.number_input("Tenure (months)", min_value=0, step=1)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", option)
    online_backup = st.selectbox("Online Backup", option)
    device_protection = st.selectbox("Device Protection", option)
    tech_support = st.selectbox("Tech Support", option)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)

    total_charges = tenure * monthly_charges
    st.markdown(f"**Total Charges: ₹{total_charges:.2f}**")

    predict_btn = st.button("Predict")


# ================== INPUT ==================

input_data = {
    "tenure": tenure,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_data])
input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')


# ================== OUTPUT ==================

st.subheader("Prediction Result")

if predict_btn:
    prediction, probability, shap_values, preprocessor = predict_churn_and_shap_analysis(input_df)

    # -------- Prediction --------
    if prediction == 1:
        st.error(f"⚠️ Customer will churn ({probability:.2%} probability)")
    else:
        st.success(f"✅ Customer will stay ({1 - probability:.2%} confidence)")

    # -------- SHAP TABLE --------
    st.subheader("SHAP Explanation")

    feature_names = get_feature_names(preprocessor)
    clean_names = [clean_feature_name(f) for f in feature_names]

    shap_value = shap_values.values[0]

    shap_df = pd.DataFrame({
        "Feature": clean_names,
        "Impact": shap_value
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.dataframe(shap_df)

    # -------- TEXT EXPLANATION --------
    st.subheader("Why this prediction?")

    top_features = shap_df.head(5)

    for _, row in top_features.iterrows():
        if row["Impact"] > 0:
            st.write(f"🔺 **{row['Feature']}** increases churn likelihood")
        else:
            st.write(f"🔻 **{row['Feature']}** decreases churn likelihood")

    # -------- VISUAL SHAP --------
    st.subheader("Visual Explanation")

    # Inject clean names into SHAP object
    shap_values.feature_names = clean_names

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap_values[0],
        max_display=10, 
        show=False
    )
    st.pyplot(fig)

    # -------- BUSINESS INSIGHT --------
    st.subheader("AI Insight")

    if prediction == 1:
        st.warning("🚨 High churn risk detected")

        risky_features = top_features[top_features['Impact'] > 0]['Feature'].values

        if any("MonthlyCharges" in f for f in risky_features):
            st.write("👉 Customer may be price sensitive")

        if any("TechSupport" in f for f in risky_features):
            st.write("👉 Offer tech support to improve retention")

        if any("Contract" in f for f in risky_features):
            st.write("👉 Suggest long-term contract plans")

    else:
        st.success("Customer is stable")