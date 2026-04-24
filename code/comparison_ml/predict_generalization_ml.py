# -*- coding: utf-8 -*-
"""
Prediction script for external generalization dataset

This script:
1) Loads the trained salt encoder and saved models
2) Loads the external generalization Excel dataset
3) Runs predictions with all saved models
4) Saves:
   - Generalization_AllModels.csv

Important:
The generalization file has no Eta column,
so this script only predicts and does NOT compute metrics.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ==============================================================================
# Feature transformation
# ==============================================================================

def transform_with_encoder(df: pd.DataFrame, salt_encoder) -> pd.DataFrame:
    df2 = df.copy()
    classes = set(salt_encoder.classes_.tolist())

    def _map(v):
        v = str(v)
        if v in classes:
            return int(salt_encoder.transform([v])[0])
        return -1

    df2["Salt"] = df2["Salt"].astype(str).apply(_map)
    return df2


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    CODE_DIR = Path(__file__).resolve().parent
    REPO_DIR = CODE_DIR.parent.parent
    DATA_DIR = REPO_DIR / "data"

    model_dir = str(CODE_DIR / "ml_models")
    base_out_dir = str(CODE_DIR / "ml_outputs")
    test_out_dir = os.path.join(base_out_dir, "test")
    generalization_out_dir = os.path.join(base_out_dir, "generalization")

    os.makedirs(generalization_out_dir, exist_ok=True)

    generalization_xlsx = str(DATA_DIR / "Rheo_Generalization_Data.xlsx")
    encoder_path = os.path.join(model_dir, "salt_encoder.pkl")
    metrics_test_csv = os.path.join(test_out_dir, "metrics_test.csv")

    if not os.path.exists(generalization_xlsx):
        raise FileNotFoundError(f"Generalization file not found: {generalization_xlsx}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    if not os.path.exists(metrics_test_csv):
        raise FileNotFoundError(f"Model ranking file not found: {metrics_test_csv}")

    df_metrics_test = pd.read_csv(metrics_test_csv).sort_values("R2", ascending=False).reset_index(drop=True)
    model_names = df_metrics_test["Model"].tolist()

    if len(model_names) == 0:
        raise ValueError(f"No model records found in {metrics_test_csv}.")

    print("Loading generalization dataset...")
    df_gen = pd.read_excel(generalization_xlsx, header=1)
    df_gen.columns = ["T", "Salt", "Cs", "fs", "Cp", "Gamma"]

    df_gen = df_gen.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma"]).copy()
    df_gen = df_gen[df_gen["Gamma"] > 0].copy()

    print(f"Number of generalization samples: {len(df_gen)}")
    if len(df_gen) == 0:
        raise ValueError("No valid samples found in the generalization file.")

    salt_encoder = joblib.load(encoder_path)

    feat_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma"]
    df_gen_all = df_gen.copy()

    X_gen_df = transform_with_encoder(df_gen[feat_cols].copy(), salt_encoder)

    for model_name in model_names:
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        payload = joblib.load(model_path)

        best_model = payload["best_model"]
        use_log_target = bool(payload["use_log_target"])

        pred_gen_fit = best_model.predict(X_gen_df)

        if use_log_target:
            pred_gen = np.power(10.0, pred_gen_fit)
        else:
            pred_gen = pred_gen_fit

        pred_gen = np.maximum(pred_gen, 1e-12)
        df_gen_all[f"Eta_pred_{model_name}"] = pred_gen

    generalization_csv = os.path.join(generalization_out_dir, "Generalization_AllModels.csv")
    df_gen_all.to_csv(generalization_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("Generalization prediction done.")
    print("No metrics were computed because the generalization file has no Eta column.")
    print(f"Saved to: {generalization_csv}")