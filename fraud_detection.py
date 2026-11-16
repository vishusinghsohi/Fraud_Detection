import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Fraud Detection App")
st.markdown("Please enter the transaction details and click the 'Predict' button to check if the transaction is fraudulent or not.")

st.divider()

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
)
step = st.number_input("Step", min_value=1, value=1)
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "step": [step],
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    # Add missing engineered features
    input_data["balanceDiffOrig"] = (
        input_data["oldbalanceOrg"] - input_data["newbalanceOrig"]
    )

    input_data["balanceDiffDest"] = (
        input_data["newbalanceDest"] - input_data["oldbalanceDest"]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction: '{int(prediction)}'")

    if prediction == 1:
        st.error("The transaction is predicted to be fraud.")
    else:
        st.success("The transaction is predicted to be legitimate.")