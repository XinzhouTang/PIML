# -*- coding: utf-8 -*-
"""
用已训练好的 RheoHybridModel.joblib
直接验证 Rheo_GT100_Validation.csv
"""

import __main__
from train_piml import RheoConfig, RheoHybridModel

# 为 joblib 反序列化提供类定义
__main__.RheoConfig = RheoConfig
__main__.RheoHybridModel = RheoHybridModel

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calc_mape(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100.0)


if __name__ == "__main__":
    model_path = "RheoHybridModel.joblib"
    val_path = "Rheo_GT100_Validation.csv"
    out_path = "Rheo_GT100_Validation_Result.csv"

    print(">>> 加载模型...")
    model = RheoHybridModel.load(model_path, verbose=True)

    print(">>> 读取验证集...")
    df_val = pd.read_csv(val_path)

    required_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]
    for c in required_cols:
        if c not in df_val.columns:
            raise ValueError(f"验证文件缺少必要列: {c}")

    # 只保留 T > 100 的样本
    df_val = df_val[df_val["T"] > 100].copy()
    print(f">>> 高温验证样本数: {len(df_val)}")

    if len(df_val) == 0:
        raise ValueError("Rheo_GT100_Validation.csv 里没有 T > 100 的数据。")

    print(">>> 开始预测...")
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

    print("\n>>> 验证结果（T > 100℃）")
    print(f"MSE      = {mse:.6e}")
    print(f"R2       = {r2:.6f}")
    print(f"MAE      = {mae:.6e}")
    print(f"MAPE(%)  = {mape:.4f}")
    print(f"Log-R2   = {log_r2:.6f}")
    print(f"Log-MAE  = {log_mae:.6f}")
    print(f"Log-RMSE = {log_rmse:.6f}")

    print(f"\n>>> 已保存结果文件: {out_path}")