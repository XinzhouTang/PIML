# -*- coding: utf-8 -*-
"""
Validate Rheo_GT100_Validation.csv
using the trained RheoHybridModel.joblib
------------------------------------------------
This script is intentionally separate from training.

Workflow:
- Load the trained hybrid model
- Read the held-out high-temperature validation file
- Keep only T > 100°C samples
- Predict viscosity
- Compute validation metrics
- Save the validation result file
"""

import __main__
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from train_piml import RheoConfig, RheoHybridModel

__main__.RheoConfig = RheoConfig
__main__.RheoHybridModel = RheoHybridModel

# ==============================================================================
# Paths
# ==============================================================================
CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CODE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SPLIT_DIR = OUTPUT_DIR / "splits"
RESULT_DIR = OUTPUT_DIR / "results"

# Ensure output folders exist
for d in [OUTPUT_DIR, MODEL_DIR, SPLIT_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Utility function
# ==============================================================================

def calc_mape(y_true, y_pred, eps=1e-12):
    """
    Compute mean absolute percentage error (MAPE).

    eps is added to the denominator to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100.0)


# ==============================================================================
# Main script
# ==============================================================================

if __name__ == "__main__":
    # Input/output files
    model_path = MODEL_DIR / "RheoHybridModel.joblib"
    val_path = SPLIT_DIR / "Rheo_GT100_Validation.csv"
    out_path = RESULT_DIR / "Rheo_GT100_Validation_Result.csv"

    print(">>> Loading model ...")
    print(f">>> Model file: {model_path}")

    # Load the trained model
    model = RheoHybridModel.load(model_path, verbose=True)

    print(">>> Reading validation dataset ...")
    print(f">>> Validation file: {val_path}")

    # Read validation data created during training
    df_val = pd.read_csv(val_path)

    # Check required columns
    required_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]
    for c in required_cols:
        if c not in df_val.columns:
            raise ValueError(f"Validation file is missing a required column: {c}")

    # Keep only high-temperature samples
    # This protects against accidental extra rows in the validation file
    df_val = df_val[df_val["T"] > 100].copy()
    print(f">>> Number of high-temperature validation samples: {len(df_val)}")

    if len(df_val) == 0:
        raise ValueError("No samples with T > 100°C were found in the validation file.")

    print(">>> Running prediction ...")

    # Predict viscosity using the trained model
    eta_pred = model.predict(
        df_val[["T", "Salt", "Cs", "fs", "Cp", "Gamma"]].copy()
    )

    # Ground truth and prediction
    y_true = df_val["Eta"].values.astype(float)
    y_pred = eta_pred.astype(float)

    # Standard metrics in original scale
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = calc_mape(y_true, y_pred)

    # Metrics in log scale
    log_true = np.log10(y_true + 1e-12)
    log_pred = np.log10(y_pred + 1e-12)
    log_r2 = r2_score(log_true, log_pred)
    log_mae = float(np.mean(np.abs(log_pred - log_true)))
    log_rmse = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))

    # Save row-wise prediction results
    df_out = df_val.copy()
    df_out["Eta_pred"] = y_pred
    df_out["Eta_abs_err"] = np.abs(df_out["Eta_pred"] - df_out["Eta"])

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Print summary
    print("\n>>> Validation results (T > 100°C)")
    print(f"MSE      = {mse:.6e}")
    print(f"R2       = {r2:.6f}")
    print(f"MAE      = {mae:.6e}")
    print(f"MAPE (%) = {mape:.4f}")
    print(f"Log-R2   = {log_r2:.6f}")
    print(f"Log-MAE  = {log_mae:.6f}")
    print(f"Log-RMSE = {log_rmse:.6f}")

    print(f"\n>>> Validation results saved to: {out_path}")