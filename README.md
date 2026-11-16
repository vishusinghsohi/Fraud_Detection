Fraud Detection Using Machine Learning
A machine learning project that analyzes financial transactions and predicts potential fraud through an interactive Streamlit application. The project combines Python, scikit-learn, GitHub version control, and a Snowflake cloud deployment.

Project Overview
- Objective: Identify potentially fraudulent financial transactions using transaction type, amount, and account balance changes.
- Dataset: 6,362,620 transactions, including 8,213 labeled fraudulent transactions (approximately 0.13%).
- Model: Class-weighted Logistic Regression with scikit-learn preprocessing.
- Application: Streamlit interface for entering transaction details and viewing predictions.
  
Tech Stack
Python · Pandas · NumPy · scikit-learn · Joblib · Streamlit · GitHub · Snowflake

Workflow
1. Explore and clean financial transaction data.
2. Engineer sender and receiver balance-change features.
3. Scale numeric features and encode transaction types.
4. Train and evaluate a Logistic Regression pipeline.
5. Save the model with Joblib and load it in Streamlit.
6. Manage source code with GitHub and deploy the application on Snowflake.

Model Performance
Metric	Test result
Accuracy	94%
Fraud recall	~95%
Fraud precision	~2%
Fraud F1-score	0.04

Key finding: The model detects most fraudulent transactions in the test set, but its low precision means it flags many legitimate transactions. Improving precision is a priority. This is an educational prototype, not a production fraud detection system.
Project Architecture
project_architecture.png

GitHub Integration & Snowflake Deployment
- Stored the application code, notebook, model pipeline, and project documentation in GitHub.
- Configured Snowflake Git API Integration for authorized GitHub repository access.
- Created project resources in Snowflake, including database FRAUD_DETECTION_DB and schema FRAUD_SCHEMA.
- Deployed the Streamlit fraud detection application in Snowflake.
Note: Git API Integration does not itself set up automated deployment or CI/CD. A Snowflake app URL may require authorized Snowflake access.

Live App: https://frauddetection902.streamlit.app/

Dataset Source
Kaggle — Fraud Detection Dataset. The large original CSV is not stored in this GitHub repository.

Future Improvements
Improve precision, tune the decision threshold, compare alternative models, and evaluate precision–recall trade-offs.
