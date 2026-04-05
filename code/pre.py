# -*- coding: utf-8 -*-
"""
Validate Rheo_GT100_Validation.csv
using the trained RheoHybridModel.joblib
"""

import __main__
from train_piml import RheoConfig, RheoHybridModel

# Provide class definitions for joblib deserialization
__main__.RheoConfig = RheoConfig
__main__.RheoHybridModel = RheoHybridModel

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calc_mape(y_true, y_pred, eps=1e-12):
    """Compute mean absolute percentage error (MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100.0)


if __name__ == "__main__":
    model_path = "RheoHybridModel.joblib"
    val_path = "Rheo_GT100_Validation.csv"
    out_path = "Rheo_GT100_Validation_Result.csv"

    print(">>> Loading model ...")
    model = RheoHybridModel.load(model_path, verbose=True)

    print(">>> Reading validation dataset ...")
    df_val = pd.read_csv(val_path)

    required_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]
    for c in required_cols:
        if c not in df_val.columns:
            raise ValueError(f"Validation file is missing a required column: {c}")

    # Keep only samples with T > 100°C
    df_val = df_val[df_val["T"] > 100].copy()
    print(f">>> Number of high-temperature validation samples: {len(df_val)}")

    if len(df_val) == 0:
        raise ValueError("No samples with T > 100°C were found in Rheo_GT100_Validation.csv.")

    print(">>> Running prediction ...")
    eta_pred = model.predict(
        df_val[["T", "Salt", "Cs", "fs", "Cp", "Gamma"]].copy()
    )

    y_true = df_val["Eta"].values.astype(float)
    y_pred = eta_pred.astype(float)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = calc_mape(y_true, y_pred)

    log_true = np.log10(y_true + 1e-12)
    log_pred = np.log10(y_pred + 1e-12)
    log_r2 = r2_score(log_true, log_pred)
    log_mae = float(np.mean(np.abs(log_pred - log_true)))
    log_rmse = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))

    df_out = df_val.copy()
    df_out["Eta_pred"] = y_pred
    df_out["Eta_abs_err"] = np.abs(df_out["Eta_pred"] - df_out["Eta"])

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n>>> Validation results (T > 100°C)")
    print(f"MSE      = {mse:.6e}")
    print(f"R2       = {r2:.6f}")
    print(f"MAE      = {mae:.6e}")
    print(f"MAPE (%) = {mape:.4f}")
    print(f"Log-R2   = {log_r2:.6f}")
    print(f"Log-MAE  = {log_mae:.6f}")
    print(f"Log-RMSE = {log_rmse:.6f}")

    print(f"\n>>> Result file saved to: {out_path}")