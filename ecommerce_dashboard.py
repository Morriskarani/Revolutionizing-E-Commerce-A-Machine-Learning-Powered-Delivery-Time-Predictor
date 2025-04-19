# ecommerce_dashboard.py
import streamlit as st
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")  # ← Add this at the top

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("📦 E-Commerce Delivery Prediction Dashboard")
st.markdown("**Final Year Project: ML-Powered Delivery Time Predictor**")

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["📊 Overview", "📈 Visualizations", "🧠 Model Results"])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("E_Commerce.csv")
    return df

df = load_data()

# --- Pages ---
if page == "📊 Overview":
    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.markdown(f"✅ Total Rows: `{df.shape[0]}`")
    st.markdown(f"✅ Total Columns: `{df.shape[1]}`")
    st.success("Categorical columns have already been encoded for model training.")

elif page == "📈 Visualizations":
    st.header("Exploratory Visualizations")

    # Countplot of target
    st.subheader("Delivery Status Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Reached.on.Time_Y.N", data=df, ax=ax)
    ax.set_xticklabels(['On-Time (0)', 'Late (1)'])
    st.pyplot(fig)

    # Weight vs status
    st.subheader("Package Weight by Delivery Status")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Reached.on.Time_Y.N", y="Weight_in_gms", data=df, ax=ax2)
    ax2.set_xticklabels(['On-Time (0)', 'Late (1)'])
    st.pyplot(fig2)

elif page == "🧠 Model Results":
    st.header("Model Performance Comparison")

    st.markdown("Here are the **final accuracy scores** of all models you trained:")

    # Model results
    results = {
        "Model": ["Decision Tree", "Random Forest", "Logistic Regression", "KNN", "XGBoost", "MLP", "Ensemble"],
        "Accuracy": [0.6945, 0.6836, 0.6654, 0.6309, 0.6632, 0.6609, 0.6882]
    }
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # Plot
    fig3, ax3 = plt.subplots()
    sns.barplot(y="Model", x="Accuracy", data=df_results, palette="viridis", ax=ax3)
    ax3.set_xlim(0.6, 0.72)
    st.pyplot(fig3)
