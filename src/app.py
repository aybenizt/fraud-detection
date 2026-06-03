import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc

from src.config import TEST_PATH, MODEL_PATH, TARGET, FEATURES
from src.utils import get_logger


logger = get_logger("dashboard")

# Set Page Config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Custom CSS Styling for Sleek Dark Theme & Accent Colors
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #e6edf3;
    }
    .stApp header {
        background-color: rgba(15, 17, 26, 0.8);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ffcc;
    }
    .css-1r6g72q {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
    }
    h1, h2, h3 {
        color: #00ffcc;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🛡️ Fraud Detection - Model Evaluation & Threshold Tuning Dashboard")
st.markdown("""
Interact with the trained pipeline model, evaluate validation/test dataset performance, 
and tune the classification threshold based on asymmetric business costs.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Load model pipeline
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# Load dataset
@st.cache_data
def load_test_data():
    if not TEST_PATH.exists():
        return None
    return pd.read_csv(TEST_PATH)

pipeline = load_model()
test_df = load_test_data()

if pipeline is None:
    st.error(f"❌ Trained model pipeline not found at `{MODEL_PATH}`. Please run model training first.")
    st.stop()

if test_df is None:
    st.error(f"❌ Evaluation dataset not found at `{TEST_PATH}`. Please run data preparation first.")
    st.stop()

# Show dataset summary
st.sidebar.markdown("### 📊 Dataset Summary")
total_records = len(test_df)
total_fraud = int(test_df[TARGET].sum())
fraud_rate = total_fraud / total_records

st.sidebar.write(f"**Total Samples:** {total_records}")
st.sidebar.write(f"**Fraud Cases:** {total_fraud} ({fraud_rate:.4%})")
st.sidebar.write(f"**Legitimate Cases:** {total_records - total_fraud}")

# Perform inference on test data (Cached)
@st.cache_data
def get_model_predictions(_model, df):
    X = df[FEATURES]
    y = df[TARGET]
    probs = _model.predict_proba(X)[:, 1]
    return probs, y

probs, y_true = get_model_predictions(pipeline, test_df)

# Business Cost Inputs in Sidebar
st.sidebar.markdown("### 💰 Business Cost Metrics")
cost_fn = st.sidebar.number_input(
    "Cost of False Negative (Missed Fraud)", 
    min_value=0.0, value=250.0, step=10.0,
    help="Financial loss incurred for each fraudulent transaction that goes undetected (chargeback, stolen merchandise)."
)
cost_fp = st.sidebar.number_input(
    "Cost of False Positive (False Alarm)", 
    min_value=0.0, value=15.0, step=1.0,
    help="Cost of reviewing a transaction or lost customer lifetime value due to false decline friction."
)

# Threshold Selector
st.markdown("## 🔍 Interactive Decision Threshold Tuning")
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Selection")
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.02,
        help="Transactions with probability >= threshold are classified as Fraud (1)."
    )
    
    # Calculate predictions based on threshold
    y_pred = (probs >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Cost calculations
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    # Savings compared to doing nothing (classifying all as legitimate, i.e., threshold=1.0)
    cost_doing_nothing = (total_fraud * cost_fn)
    savings = cost_doing_nothing - total_cost
    
    # Render Metrics
    st.metric("Recall (Detection Rate)", f"{recall:.2%}")
    st.metric("Precision (True Declines)", f"{precision:.2%}")
    st.metric("F1-Score", f"{f1:.4f}")
    st.metric("Total Business Cost", f"${total_cost:,.2f}")
    st.metric("Net Business Savings", f"${savings:,.2f}", delta=f"${savings:,.2f}")

with col2:
    st.markdown("### Confusion Matrix & Financial Summary")
    
    # Confusion matrix visual table
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        columns=["Predicted Legitimate (0)", "Predicted Fraud (1)"],
        index=["Actual Legitimate (0)", "Actual Fraud (1)"]
    )
    
    # Custom colored confusion matrix display
    fig_cm = px.imshow(
        [[tn, fp], [fn, tp]],
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Legitimate", "Fraud"],
        y=["Legitimate", "Fraud"],
        color_continuous_scale="Viridis",
        height=320
    )
    fig_cm.update_layout(
        margin=dict(l=40, r=40, t=20, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown(f"""
    **Detailed Breakdown:**
    - **True Negatives (Legitimate Approved):** `{tn:,}`
    - **False Positives (False Alarms / Legit Declined):** `{fp:,}` (costing **${fp * cost_fp:,.2f}**)
    - **False Negatives (Missed Fraud):** `{fn:,}` (costing **${fn * cost_fn:,.2f}**)
    - **True Positives (Fraud Blocked):** `{tp:,}` (prevented **${tp * cost_fn:,.2f}** in fraud loss)
    """)

# Charting Tabs: ROC/PR Curves, Cost Curve
st.markdown("## 📈 Performance Analysis Curves")
tab1, tab2, tab3 = st.tabs(["ROC Curve", "Precision-Recall Curve", "Threshold Cost Optimizer"])

with tab1:
    st.subheader("ROC Curve (Receiver Operating Characteristic)")
    
    fpr_vals, tpr_vals, roc_thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr_vals, tpr_vals)
    
    # Find current threshold index for plotting
    current_idx = np.argmin(np.abs(roc_thresholds - threshold))
    current_fpr = fpr_vals[current_idx]
    current_tpr = tpr_vals[current_idx]
    
    fig_roc = go.Figure()
    # Diagonal baseline
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Baseline'))
    # ROC Curve
    fig_roc.add_trace(go.Scatter(x=fpr_vals, y=tpr_vals, mode='lines', line=dict(color='#00ffcc', width=3), name=f'ROC (AUC = {roc_auc:.4f})'))
    # Current Threshold Point
    fig_roc.add_trace(go.Scatter(x=[current_fpr], y=[current_tpr], mode='markers', marker=dict(size=12, color='red'), name=f'Selected Thresh ({threshold})'))
    
    fig_roc.update_layout(
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        legend=dict(x=0.6, y=0.1),
        margin=dict(l=40, r=40, t=20, b=40),
        height=450,
        template="plotly_dark"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with tab2:
    st.subheader("Precision-Recall (PR) Curve")
    
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, probs)
    # Average precision
    ap = auc(recall_vals, precision_vals)
    
    # Find current threshold index for plotting
    # pr_thresholds has 1 less element than precision_vals
    current_pr_idx = np.argmin(np.abs(np.append(pr_thresholds, 1.0) - threshold))
    current_precision = precision_vals[current_pr_idx]
    current_recall = recall_vals[current_pr_idx]
    
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', line=dict(color='#ff007f', width=3), name=f'PR Curve (AP = {ap:.4f})'))
    fig_pr.add_trace(go.Scatter(x=[current_recall], y=[current_precision], mode='markers', marker=dict(size=12, color='red'), name=f'Selected Thresh ({threshold})'))
    
    fig_pr.update_layout(
        xaxis_title="Recall (Sensitivity / Detection Rate)",
        yaxis_title="Precision (Positive Predictive Value)",
        legend=dict(x=0.1, y=0.1),
        margin=dict(l=40, r=40, t=20, b=40),
        height=450,
        template="plotly_dark"
    )
    st.plotly_chart(fig_pr, use_container_width=True)

with tab3:
    st.subheader("Financial Optimization Curve")
    st.markdown("""
    This curve calculates the total system cost at every classification threshold from 0.0 to 1.0. 
    The threshold that minimizes this cost is the **mathematically optimal threshold** for your business model.
    """)
    
    # Calculate costs for a range of thresholds
    thresholds_range = np.linspace(0.0, 1.0, 101)
    costs_range = []
    recalls_range = []
    precisions_range = []
    
    for t in thresholds_range:
        preds = (probs >= t).astype(int)
        c_tn, c_fp, c_fn, c_tp = confusion_matrix(y_true, preds).ravel()
        cost = (c_fn * cost_fn) + (c_fp * cost_fp)
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 1.0
        
        costs_range.append(cost)
        recalls_range.append(rec)
        precisions_range.append(prec)
        
    costs_range = np.array(costs_range)
    optimal_idx = np.argmin(costs_range)
    optimal_threshold = thresholds_range[optimal_idx]
    optimal_cost = costs_range[optimal_idx]
    
    fig_cost = go.Figure()
    # Cost Curve
    fig_cost.add_trace(go.Scatter(x=thresholds_range, y=costs_range, mode='lines', line=dict(color='#ff9900', width=3), name='Total System Cost'))
    # Optimal threshold marker
    fig_cost.add_trace(go.Scatter(x=[optimal_threshold], y=[optimal_cost], mode='markers', marker=dict(size=14, symbol='star', color='#00ffcc'), name=f'Optimal Threshold ({optimal_threshold:.2f})'))
    # Selected threshold marker
    selected_cost = costs_range[int(threshold * 100)] if threshold <= 1.0 else costs_range[-1]
    fig_cost.add_trace(go.Scatter(x=[threshold], y=[selected_cost], mode='markers', marker=dict(size=12, color='red'), name=f'Selected Threshold ({threshold})'))
    
    fig_cost.update_layout(
        xaxis_title="Classification Probability Threshold",
        yaxis_title="Total Business Cost ($)",
        margin=dict(l=40, r=40, t=20, b=40),
        height=450,
        template="plotly_dark"
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    
    st.success(f"💡 **Recommended Threshold:** Setting the threshold to **{optimal_threshold:.2f}** yields the lowest overall business cost of **${optimal_cost:,.2f}** (saving **${cost_doing_nothing - optimal_cost:,.2f}**).")
