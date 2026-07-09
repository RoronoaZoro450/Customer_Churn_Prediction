# Customer Churn Prediction & Retention System

Predicts customer churn, explains the prediction with SHAP, and turns that into a plain-language retention plan using an LLM.

## Live Demo
- App: https://customerchurnpredictionandretention.streamlit.app
- API docs: https://customer-churn-prediction-retention-nwue.onrender.com/docs

## What It Does
Trained on the standard Telco Customer Churn feature set (tenure, contract type, internet service and add-ons, payment method, charges).

- Takes a customer's account details as input
- Predicts churn probability using a trained Gradient Boosting classifier
- Explains the prediction with SHAP, showing which features increased or decreased churn risk
- Sends the SHAP output to Llama 3.1 8B (via LangChain) to generate the top 3 churn reasons in plain English plus a 3-part retention strategy (immediate, targeted, long-term)
- Optionally saves customer records to Supabase and retrieves them later

## Architecture
```
Streamlit UI
    |
    v
FastAPI backend (Render)
    |
    +--> sklearn pipeline + SHAP TreeExplainer --> churn probability, feature attributions
    |
    +--> Llama 3.1 8B via LangChain (consumes SHAP output) --> churn reasons + retention plan
    |
    +--> Supabase --> save / fetch customer records
```

Frontend and backend are deployed separately — the Streamlit app calls the FastAPI backend over HTTP.

## Tech Stack
- **Model**: scikit-learn pipeline (preprocessing + Gradient Boosting classifier), SHAP `TreeExplainer` for feature attribution
- **LLM**: Llama-3.1-8B-Instruct via HuggingFace Inference Endpoint, orchestrated with LangChain (`ChatPromptTemplate` + `PydanticOutputParser` for structured JSON output)
- **Backend**: FastAPI, Pydantic
- **Frontend**: Streamlit, Plotly (for the SHAP bar chart)
- **Database**: Supabase
- **Deployment**: Docker (backend, on Render), Streamlit Community Cloud (frontend)

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Returns churn prediction, probability, and raw SHAP values |
| POST | `/explain` | Returns top 3 churn reasons and a retention strategy (LLM-generated) |
| POST | `/create` | Saves a customer record to Supabase |
| GET | `/view` | Returns all saved customer records |

### Request schema (`/predict`, `/explain`, `/create`)
```json
{
  "customer_id": 1001,
  "name": "Jane Doe",
  "tenure": 12,
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5
}
```
`TotalCharges` is computed automatically as `MonthlyCharges * tenure`.

## Project Structure
```
├── Customer_Churn_Prediction.ipynb   # EDA, preprocessing, model training
├── api.py                            # FastAPI backend
├── app.py                            # Streamlit frontend
├── churn_pipeline_v1.pkl             # Trained sklearn pipeline
├── Dockerfile                        # Backend container (used for Render)
└── requirements.txt
```

## Running Locally

Clone and install:
```bash
git clone https://github.com/RoronoaZoro450/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
pip install -r requirements.txt
```

Create a `.env` file (don't commit this):
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_service_role_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

Run the backend:
```bash
uvicorn api:app --reload
```

Run the frontend:
```bash
streamlit run app.py
```
`app.py` currently points `API_URL` at the deployed Render backend — change it to `http://localhost:8000` if you want the UI to hit your local API instead.

## Docker (backend only)
```bash
docker build -t churn-api .
docker run --env-file .env -p 8000:8000 churn-api
```

## Possible Improvements
- Move `API_URL` in `app.py` to an environment variable instead of hardcoding it
- Add auth to `/create` and `/view` — both are open right now, and `/view` returns every saved customer record
- Cache `/explain` responses since the LLM call is the slowest part of the request
- Track prediction/model performance over time and add a retraining path
