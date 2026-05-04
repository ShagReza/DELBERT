"""
Minimal test: load one molecule from the example parquet and predict
with the DELBERT-WDR91 model from HuggingFace.

Run:
    python test_shay.py
"""

import pandas as pd

from inference.predict import predict

EXAMPLE_PARQUET = "data/WDR91_10-examples.parquet"
MODEL = "wanglab/delbert-wdr91"
FP_COLS = ["ECFP4", "FCFP6", "ATOMPAIR", "TOPTOR"]


def main():
    df = pd.read_parquet(EXAMPLE_PARQUET)
    print(f"Loaded {len(df)} example molecules. Using the first one.")

    print(df.columns)
    row = df.iloc[1]
    molecule = {fp: row[fp] for fp in FP_COLS}

    probs = predict(molecule, model_path=MODEL)
    print(f"\nP(active) for sample 0: {probs[0]:.4f}")


if __name__ == "__main__":
    main()
