# Customer Churn Prediction & Retention System

An end-to-end ML system that predicts telecom customer churn, explains the prediction using SHAP feature importance, and generates actionable retention strategies via an LLM.

**Live Demo:**
- Frontend (Streamlit): https://customerchurnpredictionandretention.streamlit.app
- API (FastAPI on Render): https://customer-churn-prediction-retention-nwue.onrender.com/docs

---

## What it does

A customer's data is submitted through a Streamlit UI. The FastAPI backend runs it through a trained Gradient Boosting pipeline, computes SHAP values to identify which features drove the prediction, and optionally calls Llama 3.1 via HuggingFace Inference API to generate a structured retention strategy (immediate, targeted, and long-term actions). Predictions and customer records are stored in Supabase.

---

## Architecture

```
Streamlit Frontend (Streamlit Cloud)
        |
        | HTTP (REST)
        v
FastAPI Backend (Render)
        |
   ┌────┴────┐
   |         |
ML Pipeline  Supabase DB
(GB + SHAP)
   |
LangChain → HuggingFace
(Llama 3.1-8B)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn (Gradient Boosting + ColumnTransformer pipeline) |
| Explainability | SHAP (TreeExplainer) |
| LLM | Llama 3.1-8B-Instruct via HuggingFace Inference API |
| LLM Orchestration | LangChain, PydanticOutputParser |
| Backend API | FastAPI + Pydantic |
| Database | Supabase (PostgreSQL) |
| Frontend | Streamlit + Plotly |
| Deployment | Docker (backend on Render), Streamlit Cloud (frontend) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Returns churn probability + SHAP values |
| POST | `/explain` | Returns top 3 churn reasons + LLM retention strategy |
| POST | `/create` | Saves customer record to Supabase |
| GET | `/view` | Returns all stored customers |

### Example `/predict` request

```json
{
  "customer_id": 101,
  "name": "John Doe",
  "tenure": 3,
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0
}
```

### Example `/predict` response

```json
{
  "customer_id": 101,
  "churn_prediction": "Yes",
  "probability": 0.823,
  "shap_values": {
    "tenure": -0.42,
    "Contract_Month-to-month": 0.61,
    "InternetService_Fiber optic": 0.38,
    ...
  }
}
```

---

## Input Features

| Feature | Type | Values |
|---|---|---|
| `tenure` | int | Months with company (≥ 0) |
| `InternetService` | categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | categorical | Yes / No / No internet service |
| `OnlineBackup` | categorical | Yes / No / No internet service |
| `DeviceProtection` | categorical | Yes / No / No internet service |
| `TechSupport` | categorical | Yes / No / No internet service |
| `Contract` | categorical | Month-to-month / One year / Two year |
| `PaymentMethod` | categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | float | > 0 |
| `TotalCharges` | computed | `MonthlyCharges × tenure` (auto-calculated) |

---

## Local Setup

**Prerequisites:** Python 3.10+, pip

```bash
# Clone
git clone https://github.com/RoronoaZoro450/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, HUGGINGFACEHUB_API_TOKEN
```

**Run the API:**
```bash
uvicorn api:app --reload
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

**Run the frontend** (update `API_URL` in `app.py` to `http://localhost:8000` first):
```bash
streamlit run app.py
```

---

## Docker (Backend)

```bash
docker build -t churn-api .
docker run --env-file .env -p 8000:8000 churn-api
```

---

## Environment Variables

```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

Never commit `.env` to version control.

---

## Project Structure

```
├── api.py                      # FastAPI backend (predict, explain, create, view)
├── app.py                      # Streamlit frontend
├── churn_pipeline_v1.pkl       # Trained sklearn pipeline (GB + preprocessor)
├── Customer_Churn_Prediction.ipynb  # Model training and EDA
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

---

## Dataset

Based on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The model uses 10 features selected from the full dataset.
