# Credit Risk Prediction System (German Credit Dataset)

## 📌 Overview

This project implements an end-to-end Machine Learning system to predict credit risk using the German Credit Dataset.

The objective is to classify loan applicants as:

- Good (Low Credit Risk)
- Bad (High Credit Risk)

The system evaluates financial and demographic features and estimates the probability of default, along with a confidence level and recommendation output.

---

## 🧠 Problem Statement

Financial institutions must assess the risk of loan applicants to reduce defaults and financial losses.

This project builds a predictive model that:

- Processes structured applicant data
- Applies feature encoding for categorical variables
- Trains a classification model
- Evaluates performance using ROC-AUC
- Deploys the model as an interactive web application

---

## ⚙️ Technical Implementation

### 1. Data Preprocessing
- Identified numerical and categorical features
- Applied One-Hot Encoding to categorical variables
- Used Scikit-learn `ColumnTransformer` and `Pipeline` for consistent preprocessing

### 2. Model
- Algorithm: Logistic Regression
- Target: Binary Classification (Good = 0, Bad = 1)
- Dataset Size: 1000 records
- Total Input Features: 20

### 3. Model Evaluation
- Evaluation Metric: ROC-AUC
- Achieved ROC-AUC ≈ 0.80
- Analyzed confusion matrix and classification metrics

---

## 🚀 Deployment

The trained pipeline is deployed using Streamlit.

The interactive dashboard allows users to:

- Input applicant financial and profile details
- View probability of high risk
- See model confidence level
- Receive AI-based recommendation output

---

## 📊 Application Features

- Probability-based credit risk scoring
- Risk category classification
- Model confidence estimation
- Interactive dashboard interface
- End-to-end ML pipeline integration

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Machine Learning (Logistic Regression)

---

## ▶️ How to Run the Project

1. Clone the repository:

   git clone <repository_link>

2. Install dependencies:

   pip install -r requirements.txt

3. Run the Streamlit app:

   streamlit run app.py

---

## 📈 Future Improvements

- Feature importance visualization
- Model comparison with advanced algorithms
- Threshold optimization
- Cloud deployment

---

## 👩‍💻 Author

Rohini  
Machine Learning & AI Enthusiast
