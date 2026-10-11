from pathlib import Path
import sys
import numpy as np
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from logistic import sigmoid

st.set_page_config(page_title="Transmission Line Fault Detection", page_icon="⚡", layout="centered")
st.title("Three-Phase Transmission Line Fault Detection")
st.write("Enter six instantaneous current and voltage measurements. The saved Set Q logistic regression and RBF SVM predictions are shown side by side.")

required_files = [RESULTS / "theta_set_q.npy", RESULTS / "scaling_statistics.npz", RESULTS / "best_rbf_svm.joblib"]
if not all(p.exists() for p in required_files):
    st.error("Required trained models are missing. Run data preparation, logistic regression, and SVM training first.")
    st.stop()

with st.form("measurements"):
    st.subheader("Electrical measurements")
    c1, c2 = st.columns(2)
    with c1:
        Ia = st.number_input("Ia (A)", value=0.0, format="%.6f")
        Ib = st.number_input("Ib (A)", value=0.0, format="%.6f")
        Ic = st.number_input("Ic (A)", value=0.0, format="%.6f")
    with c2:
        Va = st.number_input("Va (per unit)", value=0.0, format="%.6f")
        Vb = st.number_input("Vb (per unit)", value=0.0, format="%.6f")
        Vc = st.number_input("Vc (per unit)", value=0.0, format="%.6f")
    submitted = st.form_submit_button("Predict fault status")

if submitted:
    raw = np.array([Ia, Ib, Ic, Va, Vb, Vc], dtype=float)
    raw_q = np.concatenate([raw, raw ** 2])
    stats = np.load(RESULTS / "scaling_statistics.npz")
    x_r = (raw - stats["mean_r"]) / stats["std_r"]
    x_q = (raw_q - stats["mean_q"]) / stats["std_q"]
    x_q_design = np.concatenate([[1.0], x_q])
    theta_q = np.load(RESULTS / "theta_set_q.npy")
    p_fault = float(sigmoid(x_q_design @ theta_q))
    logistic_label = int(p_fault >= 0.5)
    rbf_model = joblib.load(RESULTS / "best_rbf_svm.joblib")
    rbf_label = int(rbf_model.predict(x_r.reshape(1, -1))[0])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Set Q Logistic Regression")
        st.metric("Estimated fault probability", f"{p_fault:.3f}")
        st.success("FAULT (1)" if logistic_label == 1 else "NO FAULT (0)")
    with col2:
        st.subheader("RBF SVM")
        st.metric("Predicted class", str(rbf_label))
        st.error("FAULT (1)" if rbf_label == 1 else "NO FAULT (0)")
    st.caption("This is an educational classifier trained on simulated dataset examples, not a certified protection relay.")
