from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = "detect dataset.csv"


def load_raw(data_dir=None):
    """Load the original fault-detection CSV without cleaning or renaming it."""
    if data_dir is None:
        data_dir = ROOT / "data"
    path = Path(data_dir) / FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download detect dataset.csv and place it in data/."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_raw()
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")
    print(df.columns.tolist())
    print(df.head(5).to_string(index=False))
