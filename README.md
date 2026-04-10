# 🚀 Customer Churn Prediction & Retention System

An end-to-end **AI-powered churn prediction system** that not only predicts customer churn but also provides **explainable insights and actionable retention strategies** using Machine Learning, SHAP, and LLMs.

---

## 🌐 Live Demo

* 🔗 Frontend (Streamlit): https://customerchurnpredictionandretention.streamlit.app
* 🔗 API (FastAPI - Render): https://customer-churn-prediction-retention-nwue.onrender.com/docs

---

## 🧠 Key Features

* 📊 **Churn Prediction** using Machine Learning (Gradient Boosting)
* 🔍 **Explainability with SHAP** (feature-level contribution analysis)
* 🤖 **AI-powered Retention Strategy** using LLM (HuggingFace)
* 🗄️ **Database Integration** with Supabase
* ⚡ **FastAPI Backend** (high-performance API)
* 🎨 **Interactive UI** built with Streamlit
* 🐳 **Dockerized Backend** for scalable deployment
* ☁️ **Cloud Deployment**

  * Backend → Render
  * Frontend → Streamlit Cloud

---

## 🏗️ System Architecture

```
User (Browser)
     ↓
Streamlit Frontend (Cloud)
     ↓
FastAPI Backend (Render)
     ↓
ML Model + SHAP + LLM
     ↓
Supabase Database
```

---

## 🛠️ Tech Stack

### 🔹 Backend

* FastAPI
* Scikit-learn
* SHAP
* LangChain
* HuggingFace (Llama 3.1)

### 🔹 Frontend

* Streamlit
* Plotly

### 🔹 Database

* Supabase

### 🔹 Deployment

* Docker
* Render
* Streamlit Cloud

---

## ⚙️ API Endpoints

| Endpoint   | Description                        |
| ---------- | ---------------------------------- |
| `/predict` | Predict customer churn             |
| `/explain` | Explain churn + retention strategy |
| `/create`  | Save customer to database          |
| `/view`    | View all customers                 |

---

## 📊 Example Output

* Churn Probability Score
* SHAP Feature Contributions
* Top 3 Reasons for Churn
* AI-generated Retention Strategy

---

## 🔐 Environment Variables

Create a `.env` file locally:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_service_role_key
HUGGINGFACEHUB_API_TOKEN=your_token
```

⚠️ Do NOT commit `.env` to GitHub.

---

## 🐳 Docker Setup (Backend)

```bash
docker build -t churn-api .
docker run --env-file .env -p 8000:8000 churn-api
```

---

## 🚀 Local Setup

```bash
# Clone repo
git clone https://github.com/your-username/your-repo.git

# Install dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn api:app --reload

# Run Streamlit
streamlit run app.py
```

---

## 💡 Future Improvements

* 🔄 Async API calls for LLM
* ⚡ Caching responses (Redis)
* 📈 Advanced dashboard analytics
* 🔐 Authentication system
* 📊 Model monitoring & retraining

---

## 👨‍💻 Author

**Your Name**

---

## ⭐ Acknowledgements

* HuggingFace
* Supabase
* Streamlit
* FastAPI

---

## 💣 Note

This project demonstrates a **production-level ML system** combining:

* Prediction
* Explainability
* AI reasoning
* Cloud deployment

---
