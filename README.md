# Customer Churn Prediction — Model Development, Validation & Deployment
## Course: Inferential Statistics
## Author: Kedarnath Nagaradone
## Project Overview

Customer churn represents one of the biggest challenges for subscription-based and telecom industries.
The objective of this project is to develop, validate, compare, and deploy a predictive model that identifies customers likely to churn, enabling proactive retention strategies.

This project applies inferential statistics and predictive modeling concepts using the Telco Customer Churn dataset.
Models were built using the CHAID (Decision Tree) and Logistic Regression algorithms, validated with accuracy and ROC-AUC metrics, and prepared for real-world deployment.


## Objectives

Develop a churn prediction model using statistical inference and machine learning.

Validate and compare CHAID and Logistic Regression models.

Evaluate models using Accuracy, ROC-AUC, and Lift/Gains Charts.

Deploy the best-performing model using Python tools (joblib, pickle).

Propose a framework for model updating and automation.


## Dataset Description

Dataset Source: Kaggle – Telco Customer Churn (blastchar)

Records: 7,043 customers
Features: 21 columns

Category	Example Columns	Description
Customer Info	gender, SeniorCitizen, Partner, Dependents	Demographic details
Account Info	tenure, Contract, PaymentMethod, PaperlessBilling	Relationship information
Billing Info	MonthlyCharges, TotalCharges	Financial metrics
Services	PhoneService, InternetService, OnlineSecurity, etc.	Telecom services subscribed
Target	Churn (Yes/No)	Indicates whether the customer left the company
⚙️ Tools and Technologies
Category	Tools Used
Language	Python 3.10
Libraries	pandas, numpy, matplotlib, seaborn, scikit-learn, pychaid, joblib
IDE	Jupyter Notebook / VS Code
Version Control	Git & GitHub
Dataset Source	Kaggle (blastchar)


## Data Preparation

Converted TotalCharges to numeric (pd.to_numeric(errors='coerce'))

Dropped missing/invalid entries (df.dropna(subset=['TotalCharges']))

Removed duplicates

Outlier detection via IQR for MonthlyCharges

One-hot encoded categorical variables using pd.get_dummies()

Target variable: Churn → {Yes: 1, No: 0}


## Exploratory Data Analysis (EDA)

Key findings:

~26–27% of customers churned.

Churn is highest among Month-to-Month contracts.

Customers with low tenure are more likely to churn.

Electronic check users exhibit higher churn.

Fiber optic internet users show greater churn tendencies.


## EDA Visuals
Figure	Description
Figure 1	Churn Distribution
Figure 2	Churn by Contract Type
Figure 3	Churn by Tenure
Figure 4	Churn by Payment Method
Figure 5	Correlation Matrix


## Model Development
 
Two models were developed and compared:

1. CHAID (Decision Tree)

Algorithm: DecisionTreeClassifier with chi-square-style splits.

Extracted decision rules for interpretability.

Top Predictors:

Tenure (0.44)

InternetService_Fiber optic (0.35)

TotalCharges, MonthlyCharges, Contract Type

2. Logistic Regression

Predicts churn probability (0–1 range).

Offers statistical inference on each variable’s contribution.


## Model Evaluation and Comparison
Model	Accuracy	ROC-AUC
CHAID (Decision Tree)	0.7754	0.8130
Logistic Regression	0.7875	0.8297


## Logistic Regression slightly outperformed CHAID.
It is selected as the final deployed model due to higher AUC and stability.

Figures:

Figure 6: ROC Curves (CHAID vs Logistic Regression)

Figure 7: Lift & Gains Chart (Logistic Regression)

Figure 8: Feature Importances (CHAID Decision Tree)


## Model Deployment
Saving and Loading Model
import joblib
joblib.dump(lr_model, "models/logistic_churn_model.joblib")
model = joblib.load("models/logistic_churn_model.joblib")

Updating Procedure

Retrain monthly with new customer data.

Track performance metrics (AUC, drift).

Automate via CI/CD tools (GitHub Actions or MLflow).

Integration Options

Batch Scoring: Daily churn score updates in CRM.

API Endpoint: Real-time churn prediction using Flask/FastAPI.


## Visuals and Results
Figure	File
1	figure1_churn_distribution.png
2	figure2_churn_by_contract.png
3	figure3_churn_by_tenure.png
4	figure4_churn_by_payment.png
5	figure5_correlation_matrix.png
6	figure6_roc_curves.png
7	figure7_lift_gains_logistic.png
8	figure8_feature_importances.png


## Installation & Usage
Prerequisites

Python ≥ 3.9

Install dependencies:

pip install -r requirements.txt

Run the notebook
jupyter notebook notebooks/CHAID.ipynb

Retrain model (optional)
python src/train_model.py --data data/Telco-Customer-Churn.csv

Predict churn
model.predict_proba(new_data)[:,1]


## Performance Summary

Final Model: Logistic Regression

Accuracy: 78.75%

ROC-AUC: 0.8297

Top Factors: Tenure, Internet Service Type, Monthly Charges

Key Insight: Customers with shorter tenure and fiber optic internet are most prone to churn.


## Future Enhancements

Introduce Random Forest / XGBoost for ensemble performance.

Add cross-validation and hyperparameter tuning.

Integrate cost-sensitive learning based on customer value.

Develop real-time churn prediction API.


## References

Kaggle: Telco Customer Churn Dataset

scikit-learn Documentation

IBM SPSS Modeler Guide (for CHAID algorithm reference)


## Contact

Author: Kedarnath Nagaradone
Email: kn3510@srmist.edu.in
