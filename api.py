from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field, computed_field 
from typing import Annotated, List, Literal
import os
import pandas as pd
import shap

from supabase import create_client, Client

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not SUPABASE_URL or not SUPABASE_KEY or not HF_TOKEN:
    raise ValueError("Missing required environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

model = joblib.load(os.path.join(os.getcwd(), "churn_pipeline_v1.pkl"))
preprocessor = model.named_steps['preprocessor']
clf = model.named_steps['classifier']
explainer = shap.TreeExplainer(clf)

numerical_features = preprocessor.named_transformers_['num'].get_feature_names_out()
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()

features_names = list(numerical_features) + list(cat_features)



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,
    huggingfacehub_api_token=HF_TOKEN
)

llm_model = ChatHuggingFace(llm=llm)
class Reason(BaseModel):
    reason: str = Field(..., description="Short title of the churn reason")
    description: str = Field(..., description="Detailed explanation of the churn reason")
    shap_value: float = Field(..., description="Impact score of the feature contributing to churn")

class RetentionStrategy(BaseModel):
    immediate_action: str = Field(..., description="Quick action to reduce churn immediately")
    targeted_action: str = Field(..., description="Personalized action based on customer profile")
    long_term_action: str = Field(..., description="Strategic action for long-term retention improvement")

class ChurnResponse(BaseModel):
    top_reasons: List[Reason] = Field(..., description="Top contributing factors for churn prediction")
    retention_strategy: RetentionStrategy = Field(..., description="Recommended actions to retain the customer")

parser = PydanticOutputParser(pydantic_object=ChurnResponse)

class ChurnInput(BaseModel):
    customer_id: Annotated[int, Field(..., ge=4, description="Unique ID (min 5 chars)")]
    name: Annotated[str, Field(..., description="Full name of the customer")]
    tenure: Annotated[int, Field(..., ge=0, description="Months stayed with company")]
    InternetService: Annotated[Literal["DSL", "Fiber optic", "No"], Field(..., description="Internet service type")]
    OnlineSecurity: Annotated[Literal["Yes", "No", "No internet service"], Field(..., description="Has online security")]
    OnlineBackup: Annotated[Literal["Yes", "No", "No internet service"], Field(..., description="Has online backup")]
    DeviceProtection: Annotated[Literal["Yes", "No", "No internet service"], Field(..., description="Has device protection")]
    TechSupport: Annotated[Literal["Yes", "No", "No internet service"], Field(..., description="Has tech support")]
    Contract: Annotated[Literal["Month-to-month", "One year", "Two year"], Field(..., description="Contract type")]
    PaymentMethod: Annotated[Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], Field(..., description="Payment method")]
    MonthlyCharges: Annotated[float, Field(..., gt=0, description="Monthly amount charged")]

    @computed_field
    @property
    def TotalCharges (self) -> float:
        return self.MonthlyCharges * self.tenure

def compute_prediction_and_shap(customers: ChurnInput):
    input_df = pd.DataFrame([{
        "tenure": customers.tenure,
        "InternetService": customers.InternetService,
        "OnlineSecurity": customers.OnlineSecurity,
        "OnlineBackup": customers.OnlineBackup,
        "DeviceProtection": customers.DeviceProtection,
        "TechSupport": customers.TechSupport,
        "Contract": customers.Contract,
        "PaymentMethod": customers.PaymentMethod,
        "MonthlyCharges": customers.MonthlyCharges,
        "TotalCharges": customers.TotalCharges
    }])

    transformed = preprocessor.transform(input_df)

    prediction = clf.predict(transformed)[0]
    shap_values = explainer.shap_values(transformed, check_additivity=False)
    shap_dict = dict(zip(features_names, shap_values[0]))
    prob = clf.predict_proba(transformed)[0][1]

    return prediction, prob, shap_dict,input_df.to_dict(orient='records')[0]

@app.get('/')
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API!"}

@app.get('/view')
def view():
    response = supabase.table("Customer").select("*").execute()
    return response.data

@app.post("/predict")
def predict_churn(customers: ChurnInput):

    prediction, prob, shap_dict, _ = compute_prediction_and_shap(customers)

    return {
        "customer_id": customers.customer_id,
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "probability": round(prob, 3),
        "shap_values": shap_dict,
    }

@app.post("/explain")
def explain_churn(customers: ChurnInput):
    
    _, prob, shap_dict , userinput = compute_prediction_and_shap(customers)

    prompt = ChatPromptTemplate.from_messages(
    [
    ("system",
    """You are a brutally honest churn analyst.

    Rules:
    - Output ONLY valid JSON
    - No extra text
    - Base reasoning ONLY on SHAP values
    - Positive SHAP = increases churn risk
    - Higher SHAP = stronger reason

    Do NOT guess anything not present in data.
    """
    ),

    ("user",
    """
    ### Customer Data
    {user_input}

    ### Model Output
    Churn Probability: {churn_probability}

    SHAP Values:
    {shap_values}

    ### Task

    1. Why is this customer churning?
    - Identify Only TOP 3 reasons using highest positive SHAP values
    - Convert feature names into clear business reasons
    - give each reason in 20-15 words 
    - with description of how it impacts churn risk

    2. Retention Strategy
    - Give 3 specific actions:
    - immediate_action (quick fix)
    - targeted_action (personalized intervention)
    - long_term_action (structural fix)

    Rules:
    - If issue is pricing → suggest offers/plan change
    - If issue is product/service → suggest improvement, NOT discounts
    - If issue is low engagement → suggest usage activation

    ### Output Format
    {format_instructions}
    """
    )
    ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm_model | parser

    input_data = {
        "user_input": userinput,
        "churn_probability": prob,
        "shap_values": shap_dict    
    }
    try:
        response = chain.invoke(input_data,config={"timeout": 10})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create")
def create_customer(customer: ChurnInput):
    existing = supabase.table("Customer").select("customer_id").eq("customer_id", customer.customer_id).execute()
    
    if len(existing.data) > 0:
        raise HTTPException(status_code=400, detail='Customer already exists')

    new_customer = customer.model_dump()
    
    res = supabase.table("Customer").insert(new_customer).execute()

    if res.data is None:
        raise HTTPException(status_code=500, detail="Insert failed")

    return {"status": "Customer saved to Supabase!"}