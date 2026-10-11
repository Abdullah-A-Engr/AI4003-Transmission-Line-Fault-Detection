from pathlib import Path
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from metrics import accuracy

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
C_VALUES = [0.1, 1, 10, 100]
GAMMA_VALUES = [0.1, 1]


def main():
    data = np.load(RESULTS / "prepared_data.npz")
    X_train, X_val = data["X_r_train"], data["X_r_val"]
    y_train, y_val = data["y_train"], data["y_val"]
    records = []
    best_linear_acc = -1.0
    best_rbf_acc = -1.0
    best_linear_settings = None
    best_rbf_settings = None

    # Task 4.1: linear SVM, exactly the required four C values.
    for C in C_VALUES:
        model = SVC(kernel="linear", C=C, cache_size=512)
        model.fit(X_train, y_train)
        val_acc = accuracy(y_val, model.predict(X_val))
        n_support = int(model.n_support_.sum())
        records.append({"Kernel": "linear", "C": C, "gamma": "—",
                        "Validation accuracy": val_acc, "Support vectors": n_support})
        print(f"linear C={C}: validation accuracy={val_acc:.4f}, support vectors={n_support}", flush=True)
        if val_acc > best_linear_acc:
            best_linear_acc = val_acc
            best_linear_settings = (C, n_support)
            joblib.dump(model, RESULTS / "best_linear_svm.joblib")
        del model
        gc.collect()

    # Task 4.2: RBF SVM, all eight required C/gamma combinations.
    for C in C_VALUES:
        for gamma in GAMMA_VALUES:
            model = SVC(kernel="rbf", C=C, gamma=gamma, cache_size=512)
            model.fit(X_train, y_train)
            val_acc = accuracy(y_val, model.predict(X_val))
            n_support = int(model.n_support_.sum())
            records.append({"Kernel": "rbf", "C": C, "gamma": gamma,
                            "Validation accuracy": val_acc, "Support vectors": n_support})
            print(f"rbf C={C}, gamma={gamma}: validation accuracy={val_acc:.4f}, support vectors={n_support}", flush=True)
            if val_acc > best_rbf_acc:
                best_rbf_acc = val_acc
                best_rbf_settings = (C, gamma, n_support)
                joblib.dump(model, RESULTS / "best_rbf_svm.joblib")
            del model
            gc.collect()

    table = pd.DataFrame(records)
    table.to_csv(RESULTS / "table3_svm_validation.csv", index=False)
    pd.DataFrame([
        {"Model": "Best linear SVM", "C": best_linear_settings[0], "gamma": "—",
         "Validation accuracy": best_linear_acc, "Support vectors": best_linear_settings[1]},
        {"Model": "Best RBF SVM", "C": best_rbf_settings[0], "gamma": best_rbf_settings[1],
         "Validation accuracy": best_rbf_acc, "Support vectors": best_rbf_settings[2]},
    ]).to_csv(RESULTS / "best_svm_settings.csv", index=False)

    # Required Task 4 figure: validation accuracy versus C on a logarithmic axis.
    fig, ax = plt.subplots(figsize=(9, 5))
    linear_rows = table[table["Kernel"] == "linear"].sort_values("C")
    rbf_rows = table[table["Kernel"] == "rbf"]
    ax.plot(linear_rows["C"].astype(float), linear_rows["Validation accuracy"], marker="o", label="Linear SVM")
    for gamma in GAMMA_VALUES:
        rows = rbf_rows[rbf_rows["gamma"].astype(float) == gamma].sort_values("C")
        ax.plot(rows["C"].astype(float), rows["Validation accuracy"], marker="o", label=f"RBF SVM (gamma={gamma})")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("SVM validation accuracy by C and kernel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure6_svm_validation_accuracy.png", dpi=160)
    plt.close(fig)
    print("Saved SVM validation table, required comparison plot, and best SVM models.")


if __name__ == "__main__":
    main()
