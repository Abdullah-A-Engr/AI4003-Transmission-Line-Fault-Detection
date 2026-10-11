from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_raw


def main():
    raw = load_raw()
    empty_columns = [c for c in raw.columns if raw[c].isna().all()]
    df = raw.drop(columns=empty_columns).rename(columns={"Output (S)": "fault"}).copy()
    for c in ["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "fault"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "fault"]).reset_index(drop=True)
    df["fault"] = df["fault"].astype(int)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # Figure 1: three currents over the first 2000 raw rows; shade fault runs.
    subset = df.iloc[:2000]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(subset))
    labels = subset["fault"].to_numpy()
    in_fault = False
    start = 0
    for i, label in enumerate(labels):
        if label == 1 and not in_fault:
            start, in_fault = i, True
        elif label == 0 and in_fault:
            ax.axvspan(start, i - 1, color="red", alpha=0.12)
            in_fault = False
    if in_fault:
        ax.axvspan(start, len(labels) - 1, color="red", alpha=0.12)
    ax.plot(x, subset["Ia"], label="Ia")
    ax.plot(x, subset["Ib"], label="Ib")
    ax.plot(x, subset["Ic"], label="Ic")
    ax.set_xlabel("Row number")
    ax.set_ylabel("Current (A)")
    ax.set_title("Ia, Ib and Ic over the first 2000 rows (fault regions shaded)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure1_currents_fault_regions.png", dpi=160)
    plt.close(fig)

    # Figure 2: Ia against Ib, coloured by fault label.
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, name in [(0, "No fault (0)"), (1, "Fault (1)")]:
        part = df[df.fault == label]
        ax.scatter(part["Ia"], part["Ib"], s=12, alpha=0.55, label=name)
    ax.set_xlabel("Ia (A)")
    ax.set_ylabel("Ib (A)")
    ax.set_title("Ia versus Ib, coloured by fault label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure2_Ia_vs_Ib.png", dpi=160)
    plt.close(fig)

    # Figure 3: Ia histograms for both classes on the same axes.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df.loc[df.fault == 0, "Ia"], bins=45, alpha=0.55, label="No fault (0)")
    ax.hist(df.loc[df.fault == 1, "Ia"], bins=45, alpha=0.55, label="Fault (1)")
    ax.set_xlabel("Ia (A)")
    ax.set_ylabel("Count")
    ax.set_title("Ia distribution for fault and no-fault rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure3_Ia_histograms.png", dpi=160)
    plt.close(fig)

    # Two or three grounded sentences per figure are stored with the results.
    notes = [
        "Figure 1: The three phase currents vary across the consecutive rows, and the shaded regions mark rows labelled as faults. The plot makes it possible to inspect whether fault labels coincide with changes in current patterns.",
        "Figure 2: Fault and no-fault rows are shown in the Ia-Ib plane. Any overlap indicates that these two current measurements alone may not cleanly separate all rows with one straight boundary.",
        "Figure 3: The two Ia distributions are overlaid to compare current values across classes. Their overlap indicates that Ia alone may not be sufficient to classify every observation correctly.",
        "Overall: A single straight line in the Ia-Ib plane may not separate the classes if their point clouds overlap or form more complex patterns. Faults can change several phase currents together, so the full current/voltage feature set and nonlinear features or kernels may improve separation."
    ]
    (out / "eda_explanations.md").write_text("# Task 2 — Exploratory plots\n\n" + "\n\n".join(notes) + "\n", encoding="utf-8")
    print("Saved the three required exploratory plots in results/.")


if __name__ == "__main__":
    main()
