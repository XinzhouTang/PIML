# -*- coding: utf-8 -*-
"""
Training script for comparison ML baselines
-------------------------------------------
Functions:
1) Read the raw training Excel file
2) Preprocess:
   - keep only rows with Eta > eta_min_keep
   - aggregate by median over (T, Salt, Cs, fs, Cp, Gamma)
3) Perform GroupShuffleSplit on the full processed training dataset
   using formulation groups (Salt, Cs, fs, Cp)
4) Train models only on the training split
5) If hyperparameter tuning is enabled, perform GroupKFold inside the training split
6) Evaluate models on train / test
7) Save:
   - preprocessed data
   - train/test split files
   - trained models
   - LabelEncoder
   - train/test prediction results and metrics
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class TrainConfig:
    CODE_DIR = Path(__file__).resolve().parent
    REPO_DIR = CODE_DIR.parent.parent
    DATA_DIR = REPO_DIR / "data"

    xlsx_path: str = str(DATA_DIR / "Rheo_Training_Data.xlsx")
    xlsx_header: int = 1

    base_out_dir: str = str(CODE_DIR / "ml_outputs")
    model_dir: str = str(CODE_DIR / "ml_models")

    random_state: int = 0
    use_log_target: bool = True
    eta_min_keep: float = 0.5

    outer_test_size: float = 0.2
    inner_cv_splits: int = 5


# ==============================================================================
# Metric functions
# ==============================================================================

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"R2": float(r2), "MSE": float(mse), "RMSE": rmse}


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100.0)


def calc_log_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
# Data preprocessing
# ==============================================================================

class DataProcessor:
    def __init__(self, eta_min_keep: float = 0.5, verbose: bool = True):
        self.eta_min_keep = eta_min_keep
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def preprocess(self, df_raw: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        self._log("  Preprocessing data...")

        df = df_raw.copy()
        df = df.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]).copy()
        df = df[df["Gamma"] > 0].copy()
        df = df[df["Eta"] > float(self.eta_min_keep)].copy()

        group_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma"]
        df_proc = (
            df.groupby(group_cols, as_index=False, sort=False)
              .agg({"Eta": "median"})
        )

        if save_path:
            df_proc.to_csv(save_path, index=False, encoding="utf-8-sig")
            self._log(f"    Preprocessed data saved to: {save_path}")

        return df_proc


# ==============================================================================
# Feature processing
# ==============================================================================

class FeatureBuilder:
    def __init__(self):
        self.le_salt = LabelEncoder()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2["Salt"] = self.le_salt.fit_transform(df2["Salt"].astype(str))
        return df2

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        classes = set(self.le_salt.classes_.tolist())

        def _map(v):
            v = str(v)
            if v in classes:
                return int(self.le_salt.transform([v])[0])
            return -1

        df2["Salt"] = df2["Salt"].astype(str).apply(_map)
        return df2


# ==============================================================================
# Grouping utilities
# ==============================================================================

def build_formula_groups(df: pd.DataFrame) -> pd.Series:
    s_salt = df["Salt"].astype(str)
    s_Cs = df["Cs"].map(lambda x: f"{float(x):.8g}")
    s_fs = df["fs"].map(lambda x: f"{float(x):.8g}")
    s_Cp = df["Cp"].map(lambda x: f"{float(x):.8g}")
    return s_salt + "|" + s_Cs + "|" + s_fs + "|" + s_Cp


def build_curve_groups(df: pd.DataFrame) -> pd.Series:
    s_formula = build_formula_groups(df)
    s_T = df["T"].map(lambda x: f"{float(x):.8g}")
    return s_formula + "|" + s_T


# ==============================================================================
# Main logic
# ==============================================================================

if __name__ == "__main__":
    cfg = TrainConfig()

    train_out_dir = os.path.join(cfg.base_out_dir, "train")
    test_out_dir = os.path.join(cfg.base_out_dir, "test")

    os.makedirs(cfg.base_out_dir, exist_ok=True)
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    print("  Reading Excel data...")
    df_raw = pd.read_excel(cfg.xlsx_path, header=cfg.xlsx_header)
    df_raw.columns = ["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]

    processor = DataProcessor(eta_min_keep=cfg.eta_min_keep, verbose=True)
    processed_path = os.path.join(cfg.base_out_dir, "Processed.csv")
    df = processor.preprocess(df_raw, save_path=processed_path)

    print(f"  Total modeling samples: {len(df)}")
    if len(df) == 0:
        raise ValueError("No valid training samples found after preprocessing.")

    groups_all = build_formula_groups(df)
    n_formula_groups = groups_all.nunique()
    print(f"  Number of formulation groups: {n_formula_groups}")

    if n_formula_groups < 2:
        raise ValueError("Not enough formulation groups for GroupShuffleSplit.")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.outer_test_size,
        random_state=cfg.random_state
    )
    tr_idx, te_idx = next(gss.split(df, groups=groups_all))

    df_train = df.iloc[tr_idx].reset_index(drop=True)
    df_test = df.iloc[te_idx].reset_index(drop=True)

    df_train["Formula_Group"] = build_formula_groups(df_train).values
    df_test["Formula_Group"] = build_formula_groups(df_test).values

    df_train["Curve_Group"] = build_curve_groups(df_train).values
    df_test["Curve_Group"] = build_curve_groups(df_test).values

    print(f"  Training samples: {len(df_train)}")
    print(f"  Testing samples : {len(df_test)}")

    train_csv = os.path.join(train_out_dir, "Train.csv")
    test_csv = os.path.join(test_out_dir, "Test.csv")

    df_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_csv, index=False, encoding="utf-8-sig")

    print(f"  Training set saved to: {train_csv}")
    print(f"  Test set saved to: {test_csv}")

    feat_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma"]
    y_col = "Eta"

    fb = FeatureBuilder()
    X_train_df = fb.fit_transform(df_train[feat_cols].copy())
    X_test_df = fb.transform(df_test[feat_cols].copy())

    y_train = df_train[y_col].values.astype(float)
    y_test = df_test[y_col].values.astype(float)

    if cfg.use_log_target:
        y_train_fit = np.log10(y_train + 1e-12)
    else:
        y_train_fit = y_train

    groups_train_formula = build_formula_groups(df_train)
    cv_splits = min(cfg.inner_cv_splits, groups_train_formula.nunique())
    if cv_splits < 2:
        raise ValueError("Not enough formulation groups in training set for GroupKFold.")

    kf = GroupKFold(n_splits=cv_splits)

    models: Dict[str, Any] = {}

    rf_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", RandomForestRegressor(random_state=cfg.random_state, n_jobs=-1))
    ])
    models["RandomForest"] = GridSearchCV(
        rf_pipe,
        param_grid={
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        cv=kf,
        scoring="r2",
        n_jobs=-1,
        return_train_score=False
    )

    dt_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", DecisionTreeRegressor(random_state=cfg.random_state))
    ])
    models["DecisionTree"] = GridSearchCV(
        dt_pipe,
        param_grid={
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [None, "sqrt", "log2"],
            "model__max_depth": [None, 10, 20, 30],
        },
        cv=kf,
        scoring="r2",
        n_jobs=-1,
        return_train_score=False
    )

    gbt_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", GradientBoostingRegressor(random_state=cfg.random_state))
    ])
    models["GBT"] = GridSearchCV(
        gbt_pipe,
        param_grid={
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        },
        cv=kf,
        scoring="r2",
        n_jobs=-1,
        return_train_score=False
    )

    xgb_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", XGBRegressor(
            random_state=cfg.random_state,
            n_jobs=-1,
            objective="reg:squarederror"
        ))
    ])
    models["XGBoost"] = GridSearchCV(
        xgb_pipe,
        param_grid={
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        },
        cv=kf,
        scoring="r2",
        n_jobs=-1,
        return_train_score=False
    )

    models["LinearRegression"] = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LinearRegression())
    ])

    encoder_path = os.path.join(cfg.model_dir, "salt_encoder.pkl")
    joblib.dump(fb.le_salt, encoder_path)
    print(f"  Encoder saved to: {encoder_path}")

    drop_aux_cols = ["Formula_Group", "Curve_Group"]

    df_train_all_base = (
        df_train.drop(columns=drop_aux_cols, errors="ignore")
        .copy()
        .rename(columns={"Eta": "Eta_true"})
    )

    df_test_all_base = (
        df_test.drop(columns=drop_aux_cols, errors="ignore")
        .copy()
        .rename(columns={"Eta": "Eta_true"})
    )

    metrics_train_rows = []
    metrics_test_rows = []
    result_cache = {}

    for model_name, model in models.items():
        print("\n" + "=" * 90)
        print(f"  Training model: {model_name}")
        print("=" * 90)

        if isinstance(model, GridSearchCV):
            model.fit(X_train_df, y_train_fit, groups=groups_train_formula)
            best_model = model.best_estimator_

            print(f"  Total param settings tested: {len(model.cv_results_['params'])}")
            print(f"  Best CV score ({model_name}): {model.best_score_:.4f}")
            print(f"  Best params   ({model_name}): {model.best_params_}")
        else:
            model.fit(X_train_df, y_train_fit)
            best_model = model
            print(f"  Best params   ({model_name}): default settings")

        pred_train_fit = best_model.predict(X_train_df)
        pred_train = np.power(10.0, pred_train_fit) if cfg.use_log_target else pred_train_fit
        pred_train = np.maximum(pred_train, 1e-12)

        pred_test_fit = best_model.predict(X_test_df)
        pred_test = np.power(10.0, pred_test_fit) if cfg.use_log_target else pred_test_fit
        pred_test = np.maximum(pred_test, 1e-12)

        train_base = calc_metrics(y_train, pred_train)
        train_mae = mean_absolute_error(y_train, pred_train)
        train_mape = calc_mape(y_train, pred_train)
        train_log = calc_log_metrics(y_train, pred_train)

        metrics_train_rows.append({
            "Model": model_name,
            "R2": train_base["R2"],
            "MSE": train_base["MSE"],
            "RMSE": train_base["RMSE"],
            "MAE": float(train_mae),
            "MAPE": float(train_mape),
            "Log_R2": train_log["Log_R2"],
            "Log_MAE": train_log["Log_MAE"],
            "Log_RMSE": train_log["Log_RMSE"],
        })

        test_base = calc_metrics(y_test, pred_test)
        test_mae = mean_absolute_error(y_test, pred_test)
        test_mape = calc_mape(y_test, pred_test)
        test_log = calc_log_metrics(y_test, pred_test)

        metrics_test_rows.append({
            "Model": model_name,
            "R2": test_base["R2"],
            "MSE": test_base["MSE"],
            "RMSE": test_base["RMSE"],
            "MAE": float(test_mae),
            "MAPE": float(test_mape),
            "Log_R2": test_log["Log_R2"],
            "Log_MAE": test_log["Log_MAE"],
            "Log_RMSE": test_log["Log_RMSE"],
        })

        print(
            f"  train: "
            f"R2={train_base['R2']:.4f} | "
            f"RMSE={train_base['RMSE']:.4e} | "
            f"MSE={train_base['MSE']:.4e} | "
            f"MAE={train_mae:.4e} | "
            f"MAPE={train_mape:.2f}%"
        )

        print(
            f"  test: "
            f"R2={test_base['R2']:.4f} | "
            f"RMSE={test_base['RMSE']:.4e} | "
            f"MSE={test_base['MSE']:.4e} | "
            f"MAE={test_mae:.4e} | "
            f"MAPE={test_mape:.2f}%"
        )

        save_payload = {
            "best_model": best_model,
            "feature_cols": feat_cols,
            "target_col": y_col,
            "use_log_target": cfg.use_log_target,
            "eta_min_keep": cfg.eta_min_keep,
            "group_definition_outer": "(Salt, Cs, fs, Cp)",
            "group_definition_inner": "(Salt, Cs, fs, Cp)",
            "curve_definition": "(Salt, Cs, fs, Cp, T)",
            "random_state": cfg.random_state,
            "model_name": model_name,
        }

        model_path = os.path.join(cfg.model_dir, f"{model_name}.pkl")
        joblib.dump(save_payload, model_path)
        print(f"  Model saved to: {model_path}")

        result_cache[model_name] = {
            "pred_train": pred_train,
            "pred_test": pred_test,
        }

    metrics_train_csv = os.path.join(train_out_dir, "metrics_train.csv")
    metrics_test_csv = os.path.join(test_out_dir, "metrics_test.csv")

    df_metrics_train = pd.DataFrame(metrics_train_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    df_metrics_test = pd.DataFrame(metrics_test_rows).sort_values("R2", ascending=False).reset_index(drop=True)

    df_metrics_train.to_csv(metrics_train_csv, index=False, encoding="utf-8-sig")
    df_metrics_test.to_csv(metrics_test_csv, index=False, encoding="utf-8-sig")

    ordered_models = df_metrics_test["Model"].tolist()

    df_train_all = df_train_all_base.copy()
    df_test_all = df_test_all_base.copy()

    for model_name in ordered_models:
        df_train_all[f"Eta_pred_{model_name}"] = result_cache[model_name]["pred_train"]
        df_test_all[f"Eta_pred_{model_name}"] = result_cache[model_name]["pred_test"]

    train_all_csv = os.path.join(train_out_dir, "Train_AllModels.csv")
    test_all_csv = os.path.join(test_out_dir, "Test_AllModels.csv")

    df_train_all.to_csv(train_all_csv, index=False, encoding="utf-8-sig")
    df_test_all.to_csv(test_all_csv, index=False, encoding="utf-8-sig")

    pretty_print_metrics_table(df_metrics_train, title="Train Metrics")
    pretty_print_metrics_table(df_metrics_test, title="Test Metrics")

    print("\n" + "=" * 90)
    print("Training done.")