"""ChurnSense AI — Streamlit customer churn prediction dashboard."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import load_dataset
from src.visualization import plot_confusion_matrix, plot_feature_importance

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
DATA_PATH = ROOT / "data" / "customer_churn.csv"

st.set_page_config(page_title="ChurnSense AI", page_icon="📉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.stApp {background: radial-gradient(circle at 15% 0%, #172554 0, #0b1220 38%, #080d18 100%); color:#e5e7eb;}
[data-testid="stSidebar"] {background:#0c1424; border-right:1px solid #233149;}
.block-container {padding-top:2rem; max-width:1400px;}
.hero {padding:28px 32px; border:1px solid #263852; border-radius:20px;
background:linear-gradient(120deg,rgba(30,64,175,.35),rgba(15,23,42,.85)); margin-bottom:22px;}
.hero h1 {font-size:2.3rem; margin:0; color:#f8fafc; font-weight:800;}
.hero p {color:#a9bad2; margin:8px 0 0;}
.metric-card {background:#111c2e; border:1px solid #27364d; padding:18px 20px; border-radius:15px;}
.metric-label {color:#91a4bd; font-size:.85rem;}
.metric-value {color:#f8fafc; font-size:1.8rem; font-weight:750; margin-top:5px;}
div[data-testid="stForm"] {background:#101a2b; border:1px solid #26364d; padding:20px; border-radius:16px;}
.stButton>button {background:linear-gradient(90deg,#2563eb,#06b6d4); color:white; border:0; border-radius:9px; font-weight:700;}
div[data-testid="stAlert"] {border-radius:12px;}
h2,h3 {color:#f1f5f9;}
small {color:#94a3b8;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_data():
    if not DATA_PATH.exists():
        return None
    return load_dataset(DATA_PATH)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>', unsafe_allow_html=True
    )


bundle = get_bundle()
data = get_data()
st.markdown(
    '<div class="hero"><h1>◈ ChurnSense AI</h1>'
    '<p>Intelligent customer churn prediction · Machine learning decision support</p></div>',
    unsafe_allow_html=True
)

if bundle is None or data is None or not METRICS_PATH.exists():
    st.error("Model artifacts are missing. From the project folder, run: `python train.py`")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text())
page = st.sidebar.radio("NAVIGATION", ["Overview", "Predict Customer", "Model Performance"])
st.sidebar.markdown("---")
st.sidebar.caption("MODEL · Tuned Random Forest")
st.sidebar.caption("DATA · Synthetic demonstration dataset")
st.sidebar.caption("ChurnSense AI · v1.0")

if page == "Overview":
    st.subheader("Customer retention overview")
    total = len(data)
    churned = int(data["churn"].sum())
    rate = churned / total * 100
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("CUSTOMERS", f"{total:,}")
    with c2: metric_card("CHURN RATE", f"{rate:.1f}%")
    with c3: metric_card("ACTIVE CUSTOMERS", f"{total-churned:,}")
    with c4: metric_card("TEST ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Churn distribution")
        counts = data["churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts.plot(kind="bar", ax=ax, color=["#38bdf8", "#f97316"], rot=0)
        ax.set_ylabel("Customers")
        ax.set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("### Contract mix")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        data["contract_type"].value_counts().plot(
            kind="barh", ax=ax, color="#818cf8"
        )
        ax.set_xlabel("Customers")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.info("Dataset is synthetic and intended for learning and demonstration—not real company data.")

elif page == "Predict Customer":
    st.subheader("Individual customer risk assessment")
    st.write("Enter a customer's subscription and service profile.")
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input("Tenure (months)", 0, 120, 12)
            monthly = st.number_input("Monthly charges (₹)", 0.0, 1000.0, 79.9, 0.1)
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        with col2:
            total = st.number_input("Total charges (₹)", 0.0, 100000.0, 958.8, 1.0)
            internet = st.selectbox("Internet service", ["Fiber", "DSL", "No"])
            tickets = st.number_input("Support tickets", 0, 30, 2)
        with col3:
            late = st.number_input("Late payments", 0, 30, 1)
            usage = st.number_input("Monthly data usage (GB)", 0.0, 2000.0, 180.0, 1.0)
            st.caption("Prediction threshold: 50%")
        submitted = st.form_submit_button("⚡ Predict Churn", use_container_width=True)
    if submitted:
        customer = pd.DataFrame([{
            "tenure_months": tenure, "monthly_charges": monthly,
            "total_charges": total, "contract_type": contract,
            "internet_service": internet, "support_tickets": tickets,
            "late_payments": late, "monthly_data_gb": usage,
        }])
        probability = float(bundle["model"].predict_proba(customer)[0, 1])
        risk = "High" if probability >= .70 else "Medium" if probability >= .40 else "Low"
        st.markdown("---")
        a, b = st.columns([1, 1])
        with a:
            st.metric("Churn probability", f"{probability*100:.1f}%")
            st.progress(probability)
        with b:
            st.metric("Risk classification", risk)
            if risk == "High":
                st.error("High risk — consider proactive retention outreach.")
            elif risk == "Medium":
                st.warning("Medium risk — monitor engagement and service experience.")
            else:
                st.success("Low risk — continue regular customer engagement.")
        st.caption("This is a model estimate, not a guarantee that a customer will leave.")

else:
    st.subheader("Model performance")
    st.caption("Metrics below are calculated on the held-out test split after training.")
    keys = [("Accuracy", "accuracy"), ("Precision", "precision"),
            ("Recall", "recall"), ("F1-score", "f1_score"), ("ROC-AUC", "roc_auc")]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, keys):
        with col: metric_card(label.upper(), f"{metrics[key]:.3f}")
    st.write("")
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("### Confusion matrix")
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]))
    with right:
        st.markdown("### Feature importance")
        model = bundle["model"]
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        names = preprocessor.get_feature_names_out()
        if hasattr(classifier, "feature_importances_"):
            st.pyplot(plot_feature_importance(names, classifier.feature_importances_))
        else:
            st.info("Feature importance is unavailable for this model.")
    st.markdown("### Model comparison")
    st.write({
        "Selected model": "Tuned Random Forest",
        "Random Forest CV ROC-AUC": round(metrics["random_forest_cv_roc_auc_mean"], 4),
        "Logistic Regression CV ROC-AUC": round(metrics["logistic_regression_cv_roc_auc_mean"], 4),
        "Best parameters": metrics["random_forest_best_params"],
        "Training rows": metrics["train_samples"],
        "Test rows": metrics["test_samples"],
    })
