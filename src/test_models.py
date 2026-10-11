from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from logistic import predict
from metrics import accuracy, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    data = np.load(RESULTS / "prepared_data.npz")
    y_test = data["y_test"]
    X_r_test = np.column_stack([np.ones(len(y_test)), data["X_r_test"]])
    X_q_test = np.column_stack([np.ones(len(y_test)), data["X_q_test"]])
    theta_r = np.load(RESULTS / "theta_set_r.npy")
    theta_q = np.load(RESULTS / "theta_set_q.npy")
    linear_svm = joblib.load(RESULTS / "best_linear_svm.joblib")
    rbf_svm = joblib.load(RESULTS / "best_rbf_svm.joblib")

    predictions = {
        "Logistic regression, Set R": predict(X_r_test, theta_r),
        "Logistic regression, Set Q": predict(X_q_test, theta_q),
        "Best linear SVM": linear_svm.predict(data["X_r_test"]),
        "Best RBF SVM": rbf_svm.predict(data["X_r_test"]),
    }
    rows = []
    matrices = []
    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred)
        tn, false_alarms = cm[0]
        missed_faults, tp = cm[1]
        rows.append({"Model": name, "Accuracy": accuracy(y_test, y_pred), "TN": int(tn),
                     "False alarms": int(false_alarms), "Missed faults": int(missed_faults), "TP": int(tp)})
        matrices.append({"Model": name, "Actual 0 / Predicted 0 (TN)": int(tn),
                         "Actual 0 / Predicted 1 (false alarm)": int(false_alarms),
                         "Actual 1 / Predicted 0 (missed fault)": int(missed_faults),
                         "Actual 1 / Predicted 1 (TP)": int(tp),
                         "Matrix row 0 [TN, false alarm]": str(cm[0].tolist()),
                         "Matrix row 1 [missed fault, TP]": str(cm[1].tolist())})
        print(f"\n{name}")
        print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
        print("Confusion matrix [[TN, false alarms], [missed faults, TP]]:")
        print(cm)
        print(f"Missed faults: {missed_faults}")
    pd.DataFrame(rows).to_csv(RESULTS / "table4_test_results.csv", index=False)
    pd.DataFrame(matrices).to_csv(RESULTS / "test_confusion_matrices.csv", index=False)
    print("\nSaved test results. Test set was used only for this final evaluation.")


if __name__ == "__main__":
    main()
