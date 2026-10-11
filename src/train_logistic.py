from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic import fit_logistic, predict, cost
from metrics import accuracy

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALPHAS = [0.01, 0.1, 1, 10]
N_ITERS = 2000


def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def main():
    data = np.load(RESULTS / "prepared_data.npz")
    y_train, y_val = data["y_train"], data["y_val"]
    summaries = []
    final_thetas = {}

    for set_name, x_train_key, x_val_key in [
        ("Set R", "X_r_train", "X_r_val"),
        ("Set Q", "X_q_train", "X_q_val"),
    ]:
        X_train = add_intercept(data[x_train_key])
        X_val = add_intercept(data[x_val_key])
        alpha_histories = {}
        decreasing_flags = {}
        for alpha in ALPHAS:
            theta, history = fit_logistic(X_train, y_train, alpha, N_ITERS)
            alpha_histories[alpha] = history
            decreasing_flags[alpha] = bool(np.all(np.diff(history) < 0))
        eligible = [alpha for alpha in ALPHAS if decreasing_flags[alpha]]
        if not eligible:
            raise RuntimeError(f"No tested learning rate had strictly decreasing cost for {set_name}.")
        chosen_alpha = max(eligible)
        theta, history = fit_logistic(X_train, y_train, chosen_alpha, N_ITERS)
        final_thetas[set_name] = theta
        train_acc = accuracy(y_train, predict(X_train, theta))
        val_acc = accuracy(y_val, predict(X_val, theta))
        summaries.append({
            "Feature set": set_name,
            "Chosen alpha": chosen_alpha,
            "Final J(theta)": float(history[-1]),
            "Training accuracy": train_acc,
            "Validation accuracy": val_acc,
            "Strictly decreasing cost": decreasing_flags[chosen_alpha],
        })

        # Required one figure per feature set, showing all four learning rates.
        fig, ax = plt.subplots(figsize=(9, 5))
        for alpha in ALPHAS:
            ax.plot(np.arange(len(alpha_histories[alpha])), alpha_histories[alpha], label=f"alpha={alpha}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost J(theta)")
        ax.set_title(f"Logistic regression cost curves — {set_name}")
        ax.legend()
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        filename = "figure4_cost_curves_set_r.png" if set_name == "Set R" else "figure5_cost_curves_set_q.png"
        fig.savefig(RESULTS / filename, dpi=160)
        plt.close(fig)

        # Save chosen theta in the assignment's requested order.
        weight_names = ["theta0", "Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
        if set_name == "Set Q":
            weight_names += ["Ia^2", "Ib^2", "Ic^2", "Va^2", "Vb^2", "Vc^2"]
        pd.DataFrame({"Weight": weight_names, "Theta": theta}).to_csv(
            RESULTS / ("theta_set_r.csv" if set_name == "Set R" else "table5_theta_set_q.csv"), index=False
        )
        np.save(RESULTS / ("theta_set_r.npy" if set_name == "Set R" else "theta_set_q.npy"), theta)
        print(f"{set_name}: alpha={chosen_alpha}, final cost={history[-1]:.8f}, train accuracy={train_acc:.4f}, validation accuracy={val_acc:.4f}")
        print("  Strictly decreasing cost by alpha:", decreasing_flags)

    pd.DataFrame(summaries).to_csv(RESULTS / "table2_logistic.csv", index=False)
    print("Saved table2_logistic.csv and required cost curves.")


if __name__ == "__main__":
    main()
