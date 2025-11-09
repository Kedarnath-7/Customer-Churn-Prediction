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


### Explanation of Model Validation and Assessment

#### 1. Model Validation: Train-Test Split

**Purpose and Importance**: The train-test split is a fundamental technique in machine learning for evaluating the performance of a model on unseen data. The core idea is to divide the available dataset into two subsets:
*   **Training Set**: Used to train the machine learning model. The model learns patterns and relationships from this data.
*   **Testing Set**: Used to evaluate the trained model's performance on data it has never seen before. This provides an unbiased estimate of how the model will perform in a real-world scenario.

**Application in this analysis**: In this notebook, the `X_encoded` (features) and `y` (target variable, Churn) data were split into training and testing sets using `train_test_split` with a `test_size=0.2` and `random_state=42`. This means 80% of the data was used for training (`X_train`, `y_train`) and 20% for testing (`X_test`, `y_test`). This approach helps in:
*   **Preventing Overfitting**: By evaluating the model on independent test data, we can detect if the model has simply memorized the training data rather than learning generalizable patterns.
*   **Estimating Generalization Performance**: The metrics calculated on the test set provide a realistic indication of how well our models (CHAID/Decision Tree and Logistic Regression) would perform on new customer data.

#### 2. Assessment Criteria

We used several metrics to evaluate the models, chosen for their relevance in churn prediction:

*   **Accuracy:**
    *   **Measures**: The proportion of total predictions that were correct. It's calculated as (True Positives + True Negatives) / Total Predictions.
    *   **Relevance for Churn Prediction**: While intuitive, accuracy can be misleading in imbalanced datasets (common in churn, where non-churners usually outnumber churners). A model predicting only the majority class (non-churn) could achieve high accuracy but fail to identify actual churners.

*   **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve):**
    *   **Measures**: The ability of a classifier to distinguish between classes (churners vs. non-churners). It represents the area under the ROC curve, which plots the True Positive Rate (Sensitivity) against the False Positive Rate (1-Specificity) across various threshold settings. An AUC of 1 is perfect, 0.5 is random.
    *   **Relevance for Churn Prediction**: ROC-AUC is crucial for imbalanced datasets. It provides a single metric that summarizes the model's performance across all possible classification thresholds, indicating how well the model ranks churners higher than non-churners. A higher AUC means better separability of classes.

*   **Lift Chart:**
    *   **Measures**: How much more likely we are to find churners by using the model compared to selecting customers randomly. It's calculated by comparing the percentage of churners identified within a certain percentage of the highest-scoring customers to the baseline percentage of churners in the entire population.
    *   **Relevance for Churn Prediction**: Lift is vital for targeted marketing or retention campaigns. If a model has a lift of 3 at the top 10% of customers, it means by targeting the top 10% customers identified by the model, we are 3 times more likely to find a churner than by randomly selecting 10% of customers. This helps prioritize interventions on high-risk customers.

*   **Gains Chart:**
    *   **Measures**: The cumulative percentage of churners identified when considering a cumulative percentage of the total population, sorted by the model's predicted probability of churn. It shows how many churners you can capture by targeting a certain proportion of your customer base.
    *   **Relevance for Churn Prediction**: Gains charts help visualize the cumulative benefit of using the model. For instance, it can show that targeting the top 30% of customers (based on model-predicted churn probability) captures 60% of all actual churners. This directly supports decision-making for resource allocation in retention efforts.

#### 3. Summary of Model Performance

Let's compare the performance of the CHAID (Decision Tree) and Logistic Regression models:

**Quantitative Metrics:**
*   **CHAID (Decision Tree) Model:**
    *   Accuracy: 0.7754
    *   ROC-AUC: 0.8130
*   **Logistic Regression Model:**
    *   Accuracy: 0.7875
    *   ROC-AUC: 0.8297

**Comparison:**
*   **Accuracy**: Logistic Regression slightly outperforms the CHAID model in terms of overall accuracy (78.75% vs 77.54%).
*   **ROC-AUC**: Logistic Regression also shows a marginally better ROC-AUC score (0.8297 vs 0.8130), indicating a slightly superior ability to differentiate between churners and non-churners across various thresholds.

**Lift and Gains Charts Interpretation:**
*   **Both models** demonstrate significant lift and gains compared to a random model, especially in the initial deciles. This means both models are effective in identifying a substantial portion of churners by targeting a small percentage of the highest-risk customers.
*   Visually inspecting the Lift and Gains charts (from previous cells), the **Logistic Regression model tends to show a slightly higher lift and steeper gains curve** in the earlier deciles compared to the CHAID model. This suggests that Logistic Regression is slightly better at concentrating the churners into the top-ranked segments of the customer base.

**Strengths and Weaknesses:**

*   **CHAID (Decision Tree) Model:**
    *   **Strengths**: Provides easily interpretable decision rules (as shown in the tree visualization and feature importances). It's great for understanding the *drivers* of churn and segmenting customers based on clear criteria. This interpretability is a major business advantage.
    *   **Weaknesses**: The performance metrics (Accuracy, ROC-AUC) are slightly lower than Logistic Regression. Decision trees can sometimes be less robust to small changes in data and might overfit if not properly pruned (though `max_depth` helped here).

*   **Logistic Regression Model:**
    *   **Strengths**: Offers competitive predictive performance (slightly higher Accuracy and ROC-AUC) and is generally more robust than a single decision tree. It provides probabilities which can be useful for ranking customers by churn risk.
    *   **Weaknesses**: While coefficients can indicate feature importance, the model's decision process is less directly interpretable than a decision tree's rules. It assumes a linear relationship between features and the log-odds of the target variable, which may not always hold true.

**Conclusion**: Both models are valuable for churn prediction. The **Logistic Regression model** shows a slightly better overall predictive power based on quantitative metrics and the visual inspection of Lift/Gains charts, making it a strong candidate for identifying churners. However, the **CHAID (Decision Tree) model's** strength lies in its interpretability, providing clear business rules that can be directly translated into targeted strategies for specific customer segments. Depending on the primary objective – purely prediction accuracy or understanding the 'why' behind churn – one model might be preferred over the other, or they could be used in conjunction.


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
