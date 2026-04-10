import streamlit as st
import requests
import plotly.express as px
import pandas as pd
API_URL = "https://customer-churn-prediction-retention-nwue.onrender.com"


# --- Custom CSS for styling ---

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📉",  
    layout="wide"
)

st.markdown("""
<style>
/* Card Container */
.card {
    background-color: #111827;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 5px solid #6366f1;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

/* Title */
.card-title {
    font-size: 18px;
    font-weight: 600;
    color: #e5e7eb;
}

/* Description */
.card-desc {
    font-size: 14px;
    color: #9ca3af;
    margin-top: 6px;
}

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: bold;
    color: #f9fafb;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Strategy box */
.strategy-box {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #10b981;
}

/* Label */
.label {
    font-weight: bold;
    color: #34d399;
}
</style>
""", unsafe_allow_html=True)


st.title("📊 Customer Churn Predictor")

# --- Input Form ---
with st.sidebar:
    st.header("Enter Customer Details")

    customer_id = st.text_input("Customer ID")
    name = st.text_input("Name")
    tenure = st.number_input("Tenure (months)", min_value=0)

    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

    monthly = st.number_input("Monthly Charges", min_value=0.0)
    
    predict_button = st.button("🔮 Predict Churn")

# --- Create Payload ---
data = {
    "customer_id": customer_id,
    "name": name,
    "tenure": tenure,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": support,
    "Contract": contract,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly
}

# --- Buttons ---

# --- Predict Button ---
if predict_button:
    res = requests.post(f"{API_URL}/predict", json=data)

    if res.status_code == 200:
        result = res.json()
        st.success(f"Prediction: {result['churn_prediction']}")
        
        st.metric(label="Churn Probability", value=f"{result['probability']*100:.2f}%", delta="High", delta_color="inverse")

        shap_items = list(result["shap_values"].items())
        df_shap = pd.DataFrame(shap_items, columns=['Feature', 'SHAP Value'])
        

        df_shap = df_shap.sort_values(by='SHAP Value', ascending=True)
            

        df_shap['Color'] = df_shap['SHAP Value'].apply(
            lambda x: 'Increases Risk' if x > 0 else 'Reduces Risk'
        )


        fig = px.bar(
            df_shap, 
            x='SHAP Value', 
            y='Feature', 
            orientation='h',
            color='Color',
            color_discrete_map={'Increases Risk': '#ef553b', 'Reduces Risk': '#636efa'},
            title="<b>Why is this customer churning?</b>",
            height=600 
        )

        st.plotly_chart(fig, use_container_width=True)


    else:
        st.error(f"Error: {res.text}")

# --- Explain Button ---
if st.button("Recommend Retention Actions"):
    res = requests.post(f"{API_URL}/explain", json=data)

    if res.status_code == 200:
        explanation = res.json()

        
        st.markdown('<div class="section-header">⚠️ Reasons for Churn</div>', unsafe_allow_html=True)

        for index, reason in enumerate(explanation["top_reasons"]):
            st.markdown(f"""
            <div class="card">
                <div class="card-title">{index + 1}. {reason['reason']}</div>
                <div class="card-desc">{reason['description']}</div>
            </div>
            """, unsafe_allow_html=True)

        
        st.markdown('<div class="section-header">💡 Retention Strategy</div>', unsafe_allow_html=True)

        strategy = explanation["retention_strategy"]

        st.markdown(f"""
        <div class="strategy-box">
            <span class="label">Immediate Action:</span><br>
            {strategy['immediate_action']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="strategy-box">
            <span class="label">Short-term Action:</span><br>
            {strategy['targeted_action']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="strategy-box">
            <span class="label">Long-term Action:</span><br>
            {strategy['long_term_action']}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error(f"Error: {res.text}")

# --- Save Button ---
if st.button("💾 Save Customer"):
    res = requests.post(f"{API_URL}/create", json=data)
    if res.status_code == 200:
        st.success("Customer saved to Supabase!")
    else:
        st.error(res.text)

if st.button("🔍 View All Customers"):
    res = requests.get(f"{API_URL}/view")
    if res.status_code == 200:
        customers = res.json()
        df_customers = pd.DataFrame(customers)
        st.dataframe(df_customers)
    else:
        st.error(f"Error: {res.text}")

