from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_raw

MEASUREMENT_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


def prepare_data():
    raw = load_raw()
    raw_rows, raw_columns = raw.shape
    original_columns = raw.columns.tolist()

    # Required Task 1 inspection: original column types and missing-value counts.
    inspection_rows = []
    for column in raw.columns:
        inspection_rows.append({
            "Column": column,
            "Original data type": str(raw[column].dtype),
            "Missing values": int(raw[column].isna().sum()),
            "Entire column empty": bool(raw[column].isna().all()),
            "Numeric measurement column": column in MEASUREMENT_COLUMNS and pd.api.types.is_numeric_dtype(raw[column]),
        })
    inspection_path = ROOT / "results" / "column_inspection.csv"
    inspection_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inspection_rows).to_csv(inspection_path, index=False)

    # Remove columns that contain no information (the CSV has two empty columns).
    empty_columns = [column for column in raw.columns if raw[column].isna().all()]
    df = raw.drop(columns=empty_columns).copy()

    if "Output (S)" not in df.columns:
        raise ValueError("Expected target column 'Output (S)' was not found.")
    df = df.rename(columns={"Output (S)": "fault"})

    missing_before = df.isna().sum()
    for column in MEASUREMENT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["fault"] = pd.to_numeric(df["fault"], errors="coerce")

    # Missing rows are not silently imputed: report them and remove incomplete rows.
    missing_after_conversion = df.isna().sum()
    rows_before_drop = len(df)
    df = df.dropna(subset=MEASUREMENT_COLUMNS + ["fault"]).copy()
    df["fault"] = df["fault"].astype(int)
    if not set(df["fault"].unique()).issubset({0, 1}):
        raise ValueError("The target must contain only binary labels 0 and 1.")

    # Exact split requested in the assignment manual.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train = int(0.6 * len(df))
    n_val = int(0.2 * len(df))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_raw = df[MEASUREMENT_COLUMNS].to_numpy(dtype=float)
    y = df["fault"].to_numpy(dtype=int)

    X_train_raw = X_raw[train_idx]
    X_val_raw = X_raw[val_idx]
    X_test_raw = X_raw[test_idx]

    # Set R: six raw measurement features, standardised using training statistics only.
    mean_r = X_train_raw.mean(axis=0)
    std_r = X_train_raw.std(axis=0)
    std_r[std_r == 0] = 1.0
    X_r = (X_raw - mean_r) / std_r

    # Set Q: raw features plus their squares, then standardise all 12 using training stats.
    X_q_raw = np.column_stack([X_raw, X_raw ** 2])
    mean_q = X_q_raw[train_idx].mean(axis=0)
    std_q = X_q_raw[train_idx].std(axis=0)
    std_q[std_q == 0] = 1.0
    X_q = (X_q_raw - mean_q) / std_q

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "prepared_data.npz",
        X_r_train=X_r[train_idx], X_r_val=X_r[val_idx], X_r_test=X_r[test_idx],
        X_q_train=X_q[train_idx], X_q_val=X_q[val_idx], X_q_test=X_q[test_idx],
        y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
        mean_r=mean_r, std_r=std_r, mean_q=mean_q, std_q=std_q,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )

    split_rows = []
    for name, split_idx in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
        split_rows.append({
            "Split": name,
            "Rows": len(split_idx),
            "Fault rows": int(y[split_idx].sum()),
            "No-fault rows": int((y[split_idx] == 0).sum()),
            "Fault fraction": float(y[split_idx].mean()),
        })

    table1 = pd.DataFrame([
        {"Item": "Raw rows", "Value": raw_rows},
        {"Item": "Raw columns", "Value": raw_columns},
        {"Item": "Columns removed", "Value": ", ".join(empty_columns) if empty_columns else "None"},
        {"Item": "Rows removed for missing/invalid target or measurements", "Value": rows_before_drop - len(df)},
        {"Item": "Missing values after cleaning", "Value": int(df.isna().sum().sum())},
        {"Item": "Fault rows", "Value": int((df.fault == 1).sum())},
        {"Item": "No-fault rows", "Value": int((df.fault == 0).sum())},
        {"Item": "Training rows", "Value": len(train_idx)},
        {"Item": "Validation rows", "Value": len(val_idx)},
        {"Item": "Test rows", "Value": len(test_idx)},
    ])
    table1.to_csv(out / "table1_data_preparation.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out / "split_class_fractions.csv", index=False)

    # Persist only the training-derived scaling statistics needed by the frontend.
    np.savez_compressed(out / "scaling_statistics.npz", mean_r=mean_r, std_r=std_r, mean_q=mean_q, std_q=std_q)

    print("=== Data preparation ===")
    print(f"Raw shape: {raw_rows} rows x {raw_columns} columns")
    print(f"Columns removed: {empty_columns if empty_columns else 'None'}")
    print("Missing values per raw column:")
    print(raw.isna().sum().to_string())
    print("Data types after conversion:")
    print(df[MEASUREMENT_COLUMNS + ["fault"]].dtypes.to_string())
    print("Numeric conversion/missing-value check after cleaning:")
    print(df[MEASUREMENT_COLUMNS + ["fault"]].isna().sum().to_string())
    print(f"Fault rows: {(df.fault == 1).sum()}")
    print(f"No-fault rows: {(df.fault == 0).sum()}")
    print(pd.DataFrame(split_rows).to_string(index=False))
    print(f"Saved prepared data to {out / 'prepared_data.npz'}")
    return df


if __name__ == "__main__":
    prepare_data()
