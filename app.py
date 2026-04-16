import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import os

class DemoModel:
        def predict(self, X):
            import numpy as np
            return np.array([[np.random.uniform(0.1, 0.9)]])

model = DemoModel()

MODEL_ID = "14Hd4bPt5vcHQ5OneFtSrxm-4gSMrx7Mf"
MODEL_PATH = "model.h5"

def download_model():
        url = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

        if not os.path.exists(MODEL_PATH):
            download_model()


# ---------------- CONFIG ----------------

st.set_page_config(
page_title="Fraud Detection System",
layout="wide",
page_icon="💳"
)

# ---------------- LOAD ASSETS ----------------

@st.cache_resource
def load_assets():
    
    
    

    
        scaler = pickle.load(open("scaler.pkl", "rb"))
        df = pd.read_csv("creditcard.csv")
        return model, scaler, df

model, scaler, df = load_assets()

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Control Panel")

mode = st.sidebar.radio("Select Mode",["🔍 Manual Input", "🎯 Demo (Real Data)"])

# ---------------- HEADER ----------------

st.markdown(""" <h1 style='text-align:center;color:#1f77b4;'>
💳 AI Credit Card Fraud Detection Dashboard </h1>
""", unsafe_allow_html=True)

st.markdown(""" <p style='text-align:center;'>
Deep Learning Model for Real-Time Fraud Detection </p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- DEMO MODE ----------------

if mode == "🎯 Demo (Real Data)":
    sample = df.sample(1).iloc[0]
    st.success("Loaded real transaction sample")
else:
    sample = None

# ---------------- INPUT SECTION ----------------

st.subheader("📥 Transaction Input")

col1, col2 = st.columns(2)

# Amount

amount = col1.number_input(
"Transaction Amount",
value=float(sample["Amount"]) if sample is not None else 0.0
)

# Features

features = []

for i in range(1, 29):
    val=st.number_input(f"V{i}",value=float(sample[f"V{i}"]) if sample is not None else 0.0 )
    features.append(val)
    st.divider()

# ---------------- PREDICTION ----------------

if st.button("🚀 Run Fraud Detection", use_container_width=True):

# Scale amount
    amount_scaled = scaler.transform([[amount]])[0][0]

# Combine data
    input_data = np.array([[amount_scaled] + features])

# Predict
    prediction = model.predict(input_data)[0][0]

# ---------------- OUTPUT ----------------
    st.subheader("📊 Analysis Result")

    colA, colB = st.columns(2)

    colA.metric("Fraud Probability", f"{prediction*100:.2f}%")

    if prediction > 0.5:
        colB.error("⚠️ FRAUD DETECTED")
    else:
        colB.success("✅ LEGITIMATE")

    st.progress(int(prediction * 100))

# ---------------- DATA INSIGHT PANEL ----------------

    st.divider()
    st.subheader("📈 Dataset Insights")

    fraud_count = df["Class"].value_counts()

    col1, col2 = st.columns(2)

    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Cases", fraud_count[1])

    st.bar_chart(fraud_count)

# ---------------- FOOTER ----------------

st.divider()
st.markdown(""" <p style='text-align:center;color:gray;'>
Enterprise AI Fraud Detection System | Deep Learning Project </p>
""", unsafe_allow_html=True)
