# Fraud Detection Using Machine Learning

An end-to-end machine learning prototype for identifying potentially fraudulent financial transactions. The project combines exploratory data analysis (EDA), feature engineering, a scikit-learn classification pipeline, and a Streamlit application for interactive predictions.

## Project overview

The goal is to investigate patterns in financial transactions and build a predictive model that flags potentially fraudulent activity. The dataset contains **6,362,620 transactions**, including **8,213 labeled fraudulent transactions** (about **0.13%** of the total), making class imbalance a key challenge.

## Problem statement

Financial institutions process large volumes of transactions, making manual detection of suspicious activity difficult. Fraud is rare in this dataset (8,213 of 6,362,620 transactions, or approximately 0.13%), so a model can appear accurate while missing fraud or producing excessive false alarms.

**Business question:** How can transaction type, amount, and changes in sender and receiver balances be used to identify potentially fraudulent transactions and support transaction screening?

**Project goal:** Explore transaction patterns, build a supervised fraud classification model, and provide an interactive interface for single-transaction predictions. This is a prototype, not a production fraud prevention system.

## How the projec addresses the problem

- Explores transaction types, amounts, fraud distribution, and account balance patterns.
- Engineers `balanceDiffOrig` and `balanceDiffDest` to capture account balance changes.
- Standardizes numerical variables and one-hot encodes transaction types.
- Uses class-weighted Logistic Regression to emphasize the rare fraud class.
- Evaluates fraud detection using precision, recall, accuracy, and a confusion matrix.
- Exposes the saved model through a Streamlit form for interactive predictions.

**Outcome:** The notebook reports approximately 94.71% accuracy and 95% fraud recall, but only about 2% fraud precision. The model catches most fraud in the test set at the cost of many false positives. Improving precision is an important next step.

## Dataset description

The uploaded **`AIML Dataset.csv`** includes **6,362,620 records** and **11 original columns**:

| Column | Meaning |
| --- | --- |
| `step` | Transaction time step |
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT) |
| `amount` | Transaction amount |
| `nameOrig` | Sender identifier |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver identifier |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Target label: 1 for fraud, 0 for legitimate |
| `isFlaggedFraud` | Pre-existing fraud flag |

**Class balance:** 8,213 fraudulent transactions (approximately 0.13%) versus 6,354,407 legitimate transactions (approximately 99.87%).

The original CSV is approximately 471 MB and is **not included in this repository**, because it exceeds GitHub's usual 100 MB per-file limit. For notebook reproduction, download the dataset from [Kaggle — Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) and save the CSV as `AIML Dataset.csv` next to `analysis_model.ipynb`.

## Tech stack

- **Python**, **Pandas**, **NumPy** — data analysis and preparation
- **Matplotlib**, **Seaborn** — visualization
- **Scikit-learn** — preprocessing, Logistic Regression, and evaluation
- **Streamlit** — interactive web interface
- **Joblib** — saved model pipeline

## Workflow

1. Explore transaction types, amounts, fraudulent transaction rates, and balance-related patterns.
2. Inspect data quality and class imbalance.
3. Engineer two balance-change features:
   - `balanceDiffOrig = oldbalanceOrg - newbalanceOrig`
   - `balanceDiffDest = newbalanceDest - oldbalanceDest`
4. Prepare data using `StandardScaler` for numeric values and `OneHotEncoder` for transaction type.
5. Train a `LogisticRegression(class_weight="balanced", max_iter=1000)` model in a scikit-learn pipeline.
6. Evaluate model predictions on a stratified 30% test split.
7. Save the trained pipeline and use it in a Streamlit app.

## Project Architecture

![Fraud Detection Project Architecture](images/fraud-detection-architecture.png)

The diagram summarizes data preparation, machine learning model development, the Streamlit application, GitHub integration, and Snowflake deployment. Git API Integration enables repository access; automatic synchronization or CI/CD requires separate configuration.

## Model results

Results shown in the provided notebook:

| Metric | Test result |
| --- | ---: |
| Accuracy | 94.71% |
| Fraud precision | 2% |
| Fraud recall | 95% |
| Fraud F1-score | 0.04 |
| True positives | 2,334 |
| False negatives | 130 |
| False positives | 100,840 |

**Interpretation:** The model detects most labeled fraudulent transactions, but it also generates a large number of false positives. Due to the highly imbalanced dataset, accuracy alone does not establish strong fraud detection performance. Future work should prioritize precision-recall trade-offs, threshold tuning, PR-AUC, and alternative models.

## Files

```text
fraud-detection-ml/
├── analysis_model.ipynb           # EDA, training and evaluation
├── fraud_detection.py            # Streamlit application
├── fraud_detection_pipeline.pkl  # Saved scikit-learn pipeline
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes large dataset/local files
└── README.md
```

## Run locally

1. Clone or download this repository.
2. Create and activate a Python environment.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the Streamlit app from the project folder:

   ```bash
   streamlit run fraud_detection.py
   ```

   If `streamlit` is not recognized, use:

   ```bash
   python -m streamlit run fraud_detection.py
   ```
   
## Streamlit Application

An interactive web application developed using Streamlit that allows users to enter financial transaction details and receive fraud predictions from the trained machine learning model.

### 1. Transaction Input Interface

Fraud Detection Prediction
![Fraud Detection Model Prediction](Fraud_detect_app_ss.png)

5. Open the local address displayed by Streamlit, enter transaction details, and select **Predict**.

The app loads `fraud_detection_pipeline.pkl`, so keep the model file in the same directory as `fraud_detection.py`. Use a compatible scikit-learn version when loading the saved pipeline; if compatibility warnings occur, retrain and re-export the model with your installed version.

## Dataset and notebook reproduction

The original dataset file is named `AIML Dataset.csv` and is **not committed to this repository** because it exceeds GitHub's standard per-file size limit. To rerun the notebook, download the CSV from [the Kaggle dataset page](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset), rename it `AIML Dataset.csv` if needed, and place it beside `analysis_model.ipynb`.

**Dataset attribution:** [Fraud Detection Dataset, published on Kaggle by amanalisiddiqui](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset). Review the source page for license and usage requirements.

## Limitations and future improvements

- **Low precision:** Many legitimate transactions are flagged; optimize for a realistic false-positive budget.
- **Validation:** Use a dedicated validation split and compare precision-recall curves and PR-AUC.
- **Model reliability:** This is an educational prototype, not a deployed financial fraud prevention system.
