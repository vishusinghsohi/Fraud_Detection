# Fraud Detection Using Machine Learning

A machine learning project that analyzes financial transactions and predicts potential fraud through an interactive **Streamlit** application. It combines **Python**, **scikit-learn**, **GitHub version control**, and **Snowflake cloud deployment**.

**🌐 [Try the Live Fraud Detection App](https://frauddetection902.streamlit.app/)**

## Project Overview

- **Objective:** Identify potentially fraudulent financial transactions using transaction type, amount, and account balance changes.
- **Dataset:** 6,362,620 transactions, including 8,213 labeled fraudulent transactions (approximately 0.13%).
- **Model:** Class-weighted Logistic Regression with scikit-learn preprocessing.
- **Application:** Streamlit interface for entering transaction details and viewing predictions.

## Tech Stack

**Python** · **Pandas** · **NumPy** · **scikit-learn** · **Joblib** · **Streamlit** · **GitHub** · **Snowflake**

## Project Workflow

1. Explore and clean financial transaction data.
2. Engineer sender and receiver balance-change features.
3. Scale numeric features and encode transaction types.
4. Train and evaluate a Logistic Regression pipeline.
5. Save the model with Joblib and load it in Streamlit.
6. Manage source code with GitHub and deploy the application on Snowflake.

## Model Performance

| Metric | Test result |
| --- | ---: |
| Accuracy | 94% |
| Fraud recall | ~95% |
| Fraud precision | ~2% |
| Fraud F1-score | 0.04 |

**Key finding:** The model detects most fraudulent transactions in the test set, but its low precision means it flags many legitimate transactions. Improving precision is a priority. **This is an educational prototype, not a production fraud detection system.**

## 🏗️ Project Architecture

![Fraud Detection Project Architecture](project_architecture.png)

*The architecture shows the flow from transaction data and model development to the Streamlit application, with GitHub repository management and Snowflake deployment.*

## GitHub Integration & Snowflake Deployment

- Stored application code, the analysis notebook, the saved model pipeline, and project documentation in GitHub.
- Configured **Snowflake Git API Integration** for authorized access to GitHub repositories.
- Created Snowflake project resources, including database `FRAUD_DETECTION_DB` and schema `FRAUD_SCHEMA`.
- Deployed the Streamlit fraud detection application in Snowflake.

> **Deployment note:** Git API Integration does not automatically set up continuous deployment or CI/CD. A Snowflake-hosted app URL may require authorized Snowflake access. The public Streamlit demo is available separately below.

### Live Application

🔗 **[Open the Fraud Detection App](https://frauddetection902.streamlit.app/)**

## Dataset Source

[Kaggle — Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)

The original CSV is too large for a standard GitHub file upload and is **not stored in this repository**.

## Future Improvements

- Reduce false positives by tuning the classification threshold.
- Evaluate precision–recall curves and PR-AUC.
- Compare Logistic Regression against alternative classification models.
- Improve deployment testing and document dependency versions for reproducibility.
- Explore a GitHub-based CI/CD workflow for future application updates.

---

**Project type:** Machine Learning · Fraud Detection · Interactive Web App · Cloud Deployment
