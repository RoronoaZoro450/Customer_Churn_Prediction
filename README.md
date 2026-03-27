# 📊 Customer Churn Prediction

## 🚀 Overview

This project predicts whether a customer is likely to churn (leave a service) using Machine Learning.
It helps businesses take **proactive retention actions** and reduce revenue loss.

---

## 🎯 Problem Statement

Customer churn is a major challenge for telecom and subscription-based companies.
The goal is to build a model that can:

* Identify high-risk customers
* Explain *why* they might churn
* Suggest retention strategies

---

## 🧠 Features

* Data preprocessing & feature engineering
* Binary encoding (Yes/No → 1/0)
* Scaled numerical features
* Machine Learning model (classification)
* SHAP-based model explainability
* Customer-level churn insights
* Streamlit web app for predictions

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* SHAP
* Streamlit
* Matplotlib / Seaborn

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│── data/
│── notebooks/
│── app.py
│── model.pkl
│── preprocessing.pkl
│── requirements.txt
│── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/RoronoaZoro450/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📈 Model Performance

<img width="444" height="206" alt="image" src="https://github.com/user-attachments/assets/853db8f2-f736-4dd8-8ea7-f6c2510cdfc5" />


---

## 🔍 Explainability (SHAP)

The model uses SHAP values to:

* Identify top factors influencing churn
* Provide human-understandable insights
* Support business decision-making

Example:

* 🔺 High monthly charges → increases churn risk
* 🔻 Long tenure → reduces churn risk

---

## 💡 Business Impact

* Early churn detection
* Targeted retention campaigns
* Improved customer lifetime value (CLV)

---

## 🔮 Future Improvements

* Add real-time data pipeline
* Deploy using Docker / Cloud
* Integrate database (PostgreSQL)

---

## 👨‍💻 Author

Nirbhay Shegale

---

## ⭐ If you like this project

Give it a star on GitHub!
