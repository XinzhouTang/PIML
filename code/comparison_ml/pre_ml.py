# -*- coding: utf-8 -*-
"""
Prediction script for high-temperature extrapolation only (T > 100°C)

This script:
1) Loads the trained salt encoder and saved models
2) Loads the high-temperature validation dataset (val_gt100)
3) Runs extrapolation predictions with all saved models
4) Saves:
   - Val_GT100_AllModels.csv
   - metrics_val_gt100.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib

from typing import Dict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ==============================================================================
# Metric utilities
# ==============================================================================

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics in the original target space."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"R2": float(r2), "MSE": float(mse), "RMSE": rmse}


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Compute mean absolute percentage error (MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100.0)


def calc_log_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics in log10 space.

    This is useful because viscosity-like targets may span several orders of magnitude.
    """
    log_true = np.log10(np.asarray(y_true, dtype=float) + 1e-12)
    log_pred = np.log10(np.asarray(y_pred, dtype=float) + 1e-12)

    log_r2 = r2_score(log_true, log_pred)
    log_mae = float(np.mean(np.abs(log_pred - log_true)))
    log_rmse = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))

    return {
        "Log_R2": float(log_r2),
        "Log_MAE": float(log_mae),
        "Log_RMSE": float(log_rmse),
    }


def pretty_print_metrics_table(df_metrics: pd.DataFrame, title: str) -> None:
    """Print a formatted metrics table sorted by R2 in descending order."""
    sub = df_metrics.copy().sort_values("R2", ascending=False)

    print("\n" + title)
    print("=" * len(title))

    show_cols = ["Model", "R2", "MSE", "RMSE", "MAE", "MAPE", "Log_R2", "Log_MAE", "Log_RMSE"]
    sub_show = sub[show_cols].copy()

    sub_show["R2"] = sub_show["R2"].map(lambda x: f"{x:.4f}")
    sub_show["MSE"] = sub_show["MSE"].map(lambda x: f"{x:.4e}")
    sub_show["RMSE"] = sub_show["RMSE"].map(lambda x: f"{x:.4e}")
    sub_show["MAE"] = sub_show["MAE"].map(lambda x: f"{x:.4e}")
    sub_show["MAPE"] = sub_show["MAPE"].map(lambda x: f"{x:.2f}%")
    sub_show["Log_R2"] = sub_show["Log_R2"].map(lambda x: f"{x:.4f}")
    sub_show["Log_MAE"] = sub_show["Log_MAE"].map(lambda x: f"{x:.4f}")
    sub_show["Log_RMSE"] = sub_show["Log_RMSE"].map(lambda x: f"{x:.4f}")

    w_model = max(14, sub_show["Model"].map(len).max())

    header = (
        f"{'Model':<{w_model}}  {'R2':>8}  {'MSE':>12}  {'RMSE':>12}  "
        f"{'MAE':>12}  {'MAPE':>10}  {'Log_R2':>8}  {'Log_MAE':>10}  {'Log_RMSE':>10}"
    )
    print(header)
    print("-" * len(header))

    for _, r in sub_show.iterrows():
        print(
            f"{r['Model']:<{w_model}}  {r['R2']:>8}  {r['MSE']:>12}  {r['RMSE']:>12}  "
            f"{r['MAE']:>12}  {r['MAPE']:>10}  {r['Log_R2']:>8}  {r['Log_MAE']:>10}  {r['Log_RMSE']:>10}"
        )


# ==============================================================================
# Feature transformation
# ==============================================================================

def transform_with_encoder(df: pd.DataFrame, salt_encoder) -> pd.DataFrame:
    """
    Transform the categorical 'Salt' column using the saved LabelEncoder.

    Any unseen salt type will be mapped to -1 so that the script can still run
    during extrapolation/inference.
    """
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
    # Directory settings
    model_dir = "./ml_models"
    base_out_dir = "./ml_outputs"
    test_out_dir = os.path.join(base_out_dir, "test")
    val_out_dir = os.path.join(base_out_dir, "val_gt100")

    # Required input files
    val_csv = os.path.join(val_out_dir, "Val_GT100.csv")
    encoder_path = os.path.join(model_dir, "salt_encoder.pkl")
    metrics_test_csv = os.path.join(test_out_dir, "metrics_test.csv")

    # Basic file existence checks
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation file not found: {val_csv}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    if not os.path.exists(metrics_test_csv):
        raise FileNotFoundError(f"Model ranking file not found: {metrics_test_csv}")

    # Use the low-temperature test ranking to define model output order
    df_metrics_test = pd.read_csv(metrics_test_csv).sort_values("R2", ascending=False).reset_index(drop=True)
    model_names = df_metrics_test["Model"].tolist()

    if len(model_names) == 0:
        raise ValueError(f"No model records found in {metrics_test_csv}.")

    print("Loading high-temperature validation set...")
    df_val = pd.read_csv(val_csv)
    print(f"Number of val_gt100 samples: {len(df_val)}")

    # Load the fitted salt encoder from training
    salt_encoder = joblib.load(encoder_path)

    feat_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma"]
    y_col = "Eta"

    # Auxiliary grouping columns are not part of the prediction output table
    drop_aux_cols = ["Formula_Group", "Curve_Group"]

    # Build output table with ground-truth values
    df_val_all = (
        df_val.drop(columns=drop_aux_cols, errors="ignore")
        .copy()
        .rename(columns={"Eta": "Eta_true"})
    )

    # Transform features before inference
    X_val_df = transform_with_encoder(df_val[feat_cols].copy(), salt_encoder)
    y_val = df_val[y_col].values.astype(float)

    metrics_rows = []

    # Predict with each saved model
    for model_name in model_names:
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        payload = joblib.load(model_path)

        best_model = payload["best_model"]
        use_log_target = bool(payload["use_log_target"])

        # Predict in training space
        pred_val_fit = best_model.predict(X_val_df)

        # Convert back to the original target scale if log-target training was used
        if use_log_target:
            pred_val = np.power(10.0, pred_val_fit)
        else:
            pred_val = pred_val_fit

        # Prevent negative or zero predictions after inverse transform
        pred_val = np.maximum(pred_val, 1e-12)

        # Save model predictions into the combined output table
        df_val_all[f"Eta_pred_{model_name}"] = pred_val

        # Evaluate model performance on the high-temperature validation set
        val_base = calc_metrics(y_val, pred_val)
        val_mae = mean_absolute_error(y_val, pred_val)
        val_mape = calc_mape(y_val, pred_val)
        val_log = calc_log_metrics(y_val, pred_val)

        metrics_rows.append({
            "Model": model_name,
            "R2": val_base["R2"],
            "MSE": val_base["MSE"],
            "RMSE": val_base["RMSE"],
            "MAE": float(val_mae),
            "MAPE": float(val_mape),
            "Log_R2": val_log["Log_R2"],
            "Log_MAE": val_log["Log_MAE"],
            "Log_RMSE": val_log["Log_RMSE"],
        })

    # Output files
    val_all_csv = os.path.join(val_out_dir, "Val_GT100_AllModels.csv")
    metrics_val_csv = os.path.join(val_out_dir, "metrics_val_gt100.csv")

    # Save prediction table
    df_val_all.to_csv(val_all_csv, index=False, encoding="utf-8-sig")

    # Save validation metrics table
    df_metrics_val = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    df_metrics_val.to_csv(metrics_val_csv, index=False, encoding="utf-8-sig")

    # Print metrics summary
    pretty_print_metrics_table(
        df_metrics_val,
        title="High-Temperature Extrapolation Metrics (val_gt100)"
    )

    print("\n" + "=" * 90)
    print("Prediction done.")