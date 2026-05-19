# -*- coding: utf-8 -*-
"""
Physics-informed machine learning model for rheological viscosity prediction.

The model combines a Carreau-Yasuda viscosity baseline with an Arrhenius
temperature shift factor. A random forest model is trained on log-space
residuals between measured viscosity and the physics-based baseline.

Workflow:
1. Read the rheology training dataset.
2. Preprocess repeated measurements.
3. Fit formulation-specific physics parameters.
4. Split the processed dataset by formulation group.
5. Train the residual-learning model.
6. Save train/test predictions, evaluation metrics, and the trained model.

External generalization prediction is performed by predict_generalization_piml.py.
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib


# ==============================================================================
# Paths
# ==============================================================================
CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent.parent
DATA_DIR = REPO_DIR / "data"
# Output files are written under code/baselines_piml1/outputs/
OUTPUT_DIR = CODE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PROCESSED_DIR = OUTPUT_DIR / "processed"
RESULT_DIR = OUTPUT_DIR / "results"

# Create output directories if required
for d in [OUTPUT_DIR, MODEL_DIR, PROCESSED_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)



# ==============================================================================
# Physics formulas
# ==============================================================================

class PhysicsFormulas:
    """Rheology model equations."""
    R = 8.314  # Gas constant, J/mol/K

    @staticmethod
    def arrhenius_log10_aT(T_K: np.ndarray, E: float, Tref_K: float) -> np.ndarray:
        """
        Arrhenius temperature shift factor in log10 form.

        log10(aT) = E / (R ln(10)) * (1/T - 1/Tref)
        """
        T_K = np.asarray(T_K, dtype=float)
        return (E / (PhysicsFormulas.R * np.log(10.0))) * ((1.0 / T_K) - (1.0 / Tref_K))

    @staticmethod
    def carreau_yasuda_eta(gamma: np.ndarray,
                           eta0: np.ndarray,
                           eta_inf: float,
                           lam: np.ndarray,
                           n: float,
                           a: float) -> np.ndarray:
        """
        Carreau-Yasuda viscosity model.

        eta(gamma) = eta_inf + (eta0 - eta_inf)
                     * (1 + (lambda * gamma)^a)^((n - 1) / a)
        """
        g = np.asarray(gamma, dtype=float) + 1e-12
        eta0 = np.asarray(eta0, dtype=float)
        lam = np.asarray(lam, dtype=float)

        base = 1.0 + (lam * g) ** a
        return eta_inf + (eta0 - eta_inf) * np.power(base, (n - 1.0) / a)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class RheoConfig:
    # Reference temperature in Celsius
    Tref_c: float = 25.0

    # Temperature-dependent eta0 exponent
    p_eta0_low: float = 1.0
    p_eta0_high: float = 0.75
    p_eta0_highT: float = 100.0

    # Default activation energy
    default_E: float = 1.05e3  # J/mol

    # Fit global activation energy from data
    fit_E_from_data: bool = True

    # Bounds for fitted activation energy
    E_min: float = 8.0e3
    E_max: float = 2.0e5

    # Apply sample weighting for temperature imbalance
    use_temp_weights: bool = True

    # Sample weighting mode
    weight_mode: str = "inv_sqrt"
    max_weight_ratio: float = 80.0

    # Include raw temperature as an ML feature
    include_raw_T_feature: bool = False

    # Minimum viscosity retained for fitting and training
    eta_min_keep: float = 0.5

    # Report formulations without fitted CY parameters
    show_missing_formula_warning: bool = True

    # Low-shear plateau identification parameters
    plateau_gamma_abs: float = 1e-2
    plateau_bottomk: int = 5

    # Residual lower bound for high-temperature low-shear samples
    res_floor_apply_Tmin: float = 100.0
    res_floor_highT: float = -0.02
    use_plateau_res_floor: bool = True

    # Residual interpolation setting for prediction
    use_residual_spike_repair: bool = True
    residual_spike_jump_threshold: float = 0.30

# ==============================================================================
# Metric functions
# ==============================================================================
def calc_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if mask.sum() == 0:
        return {
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Log_RMSE": np.nan,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    log_true = np.log10(y_true + 1e-12)
    log_pred = np.log10(y_pred + 1e-12)
    log_rmse = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Log_RMSE": log_rmse,
    }
# ==============================================================================
# Main hybrid model
# ==============================================================================

class RheoHybridModel:
    """
    Hybrid rheology model.

    The physics component provides a Carreau-Yasuda baseline with an
    Arrhenius temperature shift factor. The machine-learning component
    models log-space residuals using a random forest regressor.
    """

    def __init__(self,
                 rheo_cfg: RheoConfig = RheoConfig(),
                 verbose: bool = True):
        self.rheo_cfg = rheo_cfg
        self.verbose = verbose

        # Categorical encoder for Salt
        self.encoders: Dict[str, LabelEncoder] = {}

        # Fitted CY parameters by formulation
        self.rheo_cy_params: Dict[Tuple[str, float, float, float], Dict[str, float]] = {}

        # Formulations without available CY parameters
        self.rheo_invalid_formulations: set[Tuple[str, float, float, float]] = set()

        # Arrhenius parameters
        self.rheo_E: float = rheo_cfg.default_E
        self.rheo_Tref0: float = rheo_cfg.Tref_c + 273.15  # Kelvin

        # Random forest residual model
        self.rheo_ml = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=0
        )

        # Residual clipping limits estimated during training
        self.rheo_res_clip_low: Optional[float] = None
        self.rheo_res_clip_high: Optional[float] = None

    def _log(self, msg: str) -> None:
        """Print a message when verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def _preprocess_cat(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Encode the categorical Salt variable."""
        df_proc = df.copy()

        if is_training:
            le = LabelEncoder()
            df_proc["Salt"] = le.fit_transform(df_proc["Salt"].astype(str))
            self.encoders["Salt"] = le
        else:
            le = self.encoders.get("Salt", None)
            if le is None:
                raise RuntimeError("Model has not been trained. Missing Salt encoder.")

            classes = set(le.classes_.tolist())

            def _map(v):
                v = str(v)
                if v in classes:
                    return int(le.transform([v])[0])
                return -1

            df_proc["Salt"] = df_proc["Salt"].astype(str).apply(_map)

        return df_proc

    def _get_aT(self, T_c: np.ndarray, E: float, Tref_K: float) -> np.ndarray:
        """Convert Celsius temperatures and compute Arrhenius shift factors."""
        T_K = np.asarray(T_c, dtype=float) + 273.15
        log10_aT = PhysicsFormulas.arrhenius_log10_aT(T_K, E, Tref_K)
        return np.power(10.0, log10_aT)

    def fit_physics(self, df: pd.DataFrame) -> None:
        """
        Fit the physics component of the hybrid model.

        This includes formulation-specific Carreau-Yasuda parameters and,
        when enabled, a global activation energy.
        """
        self._log(">>> Fitting rheology physics parameters (CY + E) ...")

        df = df.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]).copy()
        df = df[df["Eta"] > float(self.rheo_cfg.eta_min_keep)].copy()

        self._fit_carreau_yasuda_params(df)

        if self.rheo_cfg.fit_E_from_data:
            E_fit = self._fit_E_from_data(df)
            if E_fit is not None:
                self.rheo_E = E_fit
                self._log(f">>> Fitted global activation energy E = {self.rheo_E:.4e} J/mol")
            else:
                self._log(f">>> Activation energy fitting failed. Using default E = {self.rheo_E:.4e} J/mol")

    def _fit_carreau_yasuda_params(self, df: pd.DataFrame) -> None:
        """Fit formulation-specific Carreau-Yasuda parameters."""
        self.rheo_cy_params = {}
        self.rheo_invalid_formulations = set()

        for (salt, Cs, fs, Cp), g in df.groupby(["Salt", "Cs", "fs", "Cp"]):
            temps = np.array(sorted(g["T"].unique()))
            if len(temps) == 0:
                continue

            key = (str(salt), float(Cs), float(fs), float(Cp))
            Tref_used = 25.0 if 25.0 in temps else float(temps[np.argmin(np.abs(temps - 25.0))])
            Tref_used_K = Tref_used + 273.15
            gref = g[g["T"] == Tref_used].copy()

            if len(gref) < 6:
                self.rheo_invalid_formulations.add(key)
                if self.rheo_cfg.show_missing_formula_warning and self.verbose:
                    print(
                        "[WARN] CY parameter fitting skipped: insufficient reference-temperature samples. "
                        f"Salt={salt}, Cs={Cs}, fs={fs}, Cp={Cp}"
                    )
                continue

            x = gref["Gamma"].values.astype(float)
            y = gref["Eta"].values.astype(float)

            p0 = [
                float(np.max(y)),
                float(max(np.min(y) * 0.1, 0.0)),
                1.0,
                0.5,
                2.0
            ]
            bounds = (
                [1e-12, 0.0, 1e-12, 0.05, 0.2],
                [1e12,  1e12, 1e6,   0.99, 10.0]
            )

            try:
                popt, _ = curve_fit(
                    lambda gamma, eta0, eta_inf, lam, n, a:
                        PhysicsFormulas.carreau_yasuda_eta(gamma, eta0, eta_inf, lam, n, a),
                    x, y, p0=p0, bounds=bounds, maxfev=40000
                )
                eta0_ref, eta_inf, lam_ref, n, a = [float(v) for v in popt]
            except Exception:
                self.rheo_invalid_formulations.add(key)
                if self.rheo_cfg.show_missing_formula_warning and self.verbose:
                    print(
                        "[WARN] CY parameter fitting skipped: nonlinear fitting did not converge. "
                        f"Salt={salt}, Cs={Cs}, fs={fs}, Cp={Cp}"
                    )
                continue

            self.rheo_cy_params[key] = {
                "Tref_K": Tref_used_K,
                "eta0_ref": eta0_ref,
                "eta_inf": eta_inf,
                "lam_ref": lam_ref,
                "n": n,
                "a": a,
            }

    def _fit_E_from_data(self, df: pd.DataFrame) -> Optional[float]:
        """Estimate the global Arrhenius activation energy from the training data."""
        Tref_target = self.rheo_cfg.Tref_c
        loga_grid = np.linspace(-3.0, 3.0, 241)

        Es: List[float] = []

        for (salt, Cs, fs, Cp), g in df.groupby(["Salt", "Cs", "fs", "Cp"]):
            g = g.copy()
            g["Gamma"] = g["Gamma"].astype(float)
            g["Eta"] = g["Eta"].astype(float)
            g = g[(g["Gamma"] > 0) & (g["Eta"] > self.rheo_cfg.eta_min_keep)]

            temps = np.array(sorted(g["T"].unique()))
            if len(temps) < 3:
                continue

            Tref_use = Tref_target if Tref_target in temps else float(temps[np.argmin(np.abs(temps - Tref_target))])
            Tref_K = Tref_use + 273.15

            gref = g[g["T"] == Tref_use].sort_values("Gamma")
            if len(gref) < 6:
                continue

            lr = np.log10(gref["Gamma"].values)
            yr = np.log10(gref["Eta"].values)

            def interp_ref(lgamma_scaled: np.ndarray) -> np.ndarray:
                return np.interp(lgamma_scaled, lr, yr, left=np.nan, right=np.nan)

            xs, ys = [], []

            for T in temps:
                if float(T) == float(Tref_use):
                    continue

                gT = g[g["T"] == T].sort_values("Gamma")
                if len(gT) < 6:
                    continue

                lT = np.log10(gT["Gamma"].values)
                yT = np.log10(gT["Eta"].values)

                best_mse = np.inf
                best_loga = None

                for loga in loga_grid:
                    yref_scaled = interp_ref(lT + loga)
                    ypred = loga + yref_scaled

                    mask = np.isfinite(ypred) & np.isfinite(yT)
                    if mask.sum() < 4:
                        continue

                    mse = np.mean((yT[mask] - ypred[mask]) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_loga = loga

                if best_loga is None:
                    continue

                log10_aT = float(best_loga)
                Tk = float(T) + 273.15
                x = (1.0 / Tk) - (1.0 / Tref_K)

                xs.append(x)
                ys.append(log10_aT)

            if len(xs) < 2:
                continue

            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)

            m, _ = np.polyfit(xs, ys, 1)
            E_i = float(m * PhysicsFormulas.R * np.log(10.0))

            if np.isfinite(E_i) and E_i > 0:
                Es.append(E_i)

        if len(Es) < 10:
            return None

        E_fit = float(np.median(Es))
        E_fit = float(np.clip(E_fit, self.rheo_cfg.E_min, self.rheo_cfg.E_max))
        return E_fit

    def _theory_eta_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Compute physics-based baseline viscosity for a batch of samples."""
        eta_out = np.full(len(df), np.nan, dtype=float)
        df_local = df.reset_index(drop=True)

        for (salt, Cs, fs, Cp), idx in df_local.groupby(["Salt_raw", "Cs", "fs", "Cp"]).groups.items():
            rows = np.array(list(idx), dtype=int)
            sub = df_local.loc[rows]

            key = (str(salt), float(Cs), float(fs), float(Cp))
            p = self.rheo_cy_params.get(key, None)

            if (key in self.rheo_invalid_formulations) or (p is None):
                if self.rheo_cfg.show_missing_formula_warning and self.verbose:
                    print(
                        "[WARN] CY parameters unavailable; Eta_Theory assigned NaN. "
                        f"Salt={salt}, Cs={Cs}, fs={fs}, Cp={Cp}"
                    )
                continue

            eta0_ref = p["eta0_ref"]
            eta_inf = p["eta_inf"]
            lam_ref = p["lam_ref"]
            n = p["n"]
            a = p["a"]
            Tref_K = p["Tref_K"]

            T_arr = sub["T"].values.astype(float)
            g_arr = sub["Gamma"].values.astype(float)

            aT = self._get_aT(T_arr, self.rheo_E, Tref_K)

            p_eta0 = np.full_like(aT, fill_value=self.rheo_cfg.p_eta0_low, dtype=float)
            p_eta0[T_arr >= self.rheo_cfg.p_eta0_highT] = self.rheo_cfg.p_eta0_high

            eta0_T = eta0_ref * np.power(aT, p_eta0)
            lam_T = lam_ref * aT

            eta_pred = PhysicsFormulas.carreau_yasuda_eta(g_arr, eta0_T, eta_inf, lam_T, n, a)
            eta_out[rows] = np.maximum(eta_pred, 1e-12)

        return eta_out

    def _compute_sample_weight(self, df_feat: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute sample weights for temperature imbalance."""
        if (not self.rheo_cfg.use_temp_weights) or (self.rheo_cfg.weight_mode == "none"):
            return None

        counts = df_feat["T"].value_counts().to_dict()
        c = df_feat["T"].map(lambda t: counts.get(t, 1)).astype(float).values

        if self.rheo_cfg.weight_mode == "inv":
            w = 1.0 / np.maximum(c, 1.0)
        elif self.rheo_cfg.weight_mode == "inv_sqrt":
            w = 1.0 / np.sqrt(np.maximum(c, 1.0))
        else:
            return None

        ratio = float(self.rheo_cfg.max_weight_ratio)
        if ratio > 1.0:
            w_min = np.min(w)
            if w_min > 0:
                w = np.clip(w, w_min, w_min * ratio)

        w = w / np.mean(w)
        return w

    def _make_plateau_mask(self, df_feat: pd.DataFrame) -> np.ndarray:
        """Identify low-shear plateau-region samples."""
        n = len(df_feat)
        if n == 0:
            return np.zeros(0, dtype=bool)

        gamma_vals = df_feat["Gamma"].astype(float).values
        mask_abs = gamma_vals <= float(self.rheo_cfg.plateau_gamma_abs)

        mask_bottomk = np.zeros(n, dtype=bool)

        group_cols = ["Salt_raw", "Cs", "fs", "Cp", "T"]
        for _, idx in df_feat.groupby(group_cols).groups.items():
            rows = np.array(list(idx), dtype=int)
            if len(rows) == 0:
                continue

            sub_gamma = df_feat.loc[rows, "Gamma"].astype(float).values
            order = np.argsort(sub_gamma)

            k = int(min(max(1, self.rheo_cfg.plateau_bottomk), len(rows)))
            chosen = rows[order[:k]]
            mask_bottomk[chosen] = True

        return mask_abs | mask_bottomk

    def preprocess_data(self, df_raw: pd.DataFrame, save_path: Path) -> pd.DataFrame:
        """
        Preprocess the rheology dataset.

        The procedure removes invalid records, aggregates repeated
        measurements by median viscosity, and saves the processed dataset.
        """
        self._log(">>> Preprocessing rheology data ...")

        df = df_raw.copy()
        df = df.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]).copy()
        df = df[df["Gamma"] > 0].copy()
        df = df[df["Eta"] > float(self.rheo_cfg.eta_min_keep)].copy()

        group_cols = ["T", "Salt", "Cs", "fs", "Cp", "Gamma"]
        df_proc = (
            df.groupby(group_cols, as_index=False, sort=False)
              .agg({"Eta": "median"})
        )

        df_proc.to_csv(save_path, index=False, encoding="utf-8-sig")
        self._log(f"    Preprocessed rheology data saved to: {save_path}")
        return df_proc

    def train(self, df_train: pd.DataFrame) -> Tuple[float, float]:
        """Train the random forest residual model."""
        self._log(">>> [Rheo] Training ...")

        df_train = df_train.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]).copy()
        df_train = df_train[df_train["Eta"] > float(self.rheo_cfg.eta_min_keep)].copy()

        df_feat = df_train.copy()
        df_feat["Salt_raw"] = df_feat["Salt"].astype(str)

        eta_theory = self._theory_eta_batch(
            df_feat[["Salt_raw", "Cs", "fs", "Cp", "T", "Gamma"]]
        )
        valid_mask = np.isfinite(eta_theory) & (eta_theory > 0)

        if not np.any(valid_mask):
            raise RuntimeError("No valid physics baseline samples are available for ML training.")

        dropped_count = int((~valid_mask).sum())
        if dropped_count > 0:
            self._log(
                f">>> [Rheo] Excluded {dropped_count} samples with unavailable physics baselines before residual-model training."
            )

        df_feat = df_feat.loc[valid_mask].reset_index(drop=True)
        eta_theory = eta_theory[valid_mask]

        df_ml = self._preprocess_cat(df_feat, is_training=True)
        df_ml["Eta_Theory"] = eta_theory
        df_ml["Log_Eta_Theory"] = np.log10(df_ml["Eta_Theory"].values + 1e-12)

        y_log = np.log10(df_ml["Eta"].values + 1e-12)
        y_res = y_log - df_ml["Log_Eta_Theory"].values

        if self.rheo_cfg.use_plateau_res_floor:
            mask_plateau = self._make_plateau_mask(df_feat)
            T_vals = df_ml["T"].astype(float).values
            mask_hiT = T_vals >= float(self.rheo_cfg.res_floor_apply_Tmin)

            mask_apply = mask_plateau & mask_hiT
            if np.any(mask_apply):
                y_res[mask_apply] = np.maximum(
                    y_res[mask_apply],
                    float(self.rheo_cfg.res_floor_highT)
                )


        aT_global = self._get_aT(
            df_ml["T"].values.astype(float),
            self.rheo_E,
            self.rheo_Tref0
        )
        df_ml["aT_global"] = aT_global

        df_ml["Gamma_Reduced"] = (
            df_ml["Gamma"].astype(float).values *
            df_ml["aT_global"].astype(float).values
        )

        base_features = [
            "Salt", "Cs", "fs", "Cp",
            "Gamma", "aT_global", "Gamma_Reduced", "Log_Eta_Theory"
        ]
        if self.rheo_cfg.include_raw_T_feature:
            base_features = ["T"] + base_features

        X = df_ml[base_features]
        sample_weight = self._compute_sample_weight(df_ml)

        if sample_weight is not None:
            self.rheo_ml.fit(X, y_res, sample_weight=sample_weight)
        else:
            self.rheo_ml.fit(X, y_res)

        res_pred = self.rheo_ml.predict(X)
        y_pred_log = df_ml["Log_Eta_Theory"].values + res_pred
        y_pred = np.power(10.0, y_pred_log)

        metrics = calc_eval_metrics(df_ml["Eta"].values, y_pred)
        return metrics

    def _repair_isolated_residual_spikes_for_prediction(
        self,
        df_info: pd.DataFrame,
        res_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate isolated residual spikes during prediction.

        The procedure is applied to the residual correction only and does not
        modify the physics-based baseline viscosity.
        """
        jump_threshold = float(
            getattr(self.rheo_cfg, "residual_spike_jump_threshold", 0.30)
        )

        df = df_info.copy().reset_index(drop=True)
        df["_row_id"] = np.arange(len(df))
        df["res_pred"] = np.asarray(res_pred, dtype=float)
        df["res_pred_fixed"] = df["res_pred"].values

        group_cols = ["T", "Salt_raw", "Cs", "fs", "Cp"]

        for _, idx in df.groupby(group_cols, sort=False).groups.items():
            idx = list(idx)

            if len(idx) < 3:
                continue

            sub = df.loc[idx].copy().sort_values("Gamma")
            rows = sub.index.to_numpy()

            gamma = sub["Gamma"].astype(float).values
            res = sub["res_pred"].astype(float).values
            res_fixed = res.copy()

            if np.any(gamma <= 0):
                continue

            log_gamma = np.log10(gamma)

            for i in range(1, len(res) - 1):
                left = res[i - 1]
                mid = res[i]
                right = res[i + 1]

                diff_left = mid - left
                diff_right = mid - right

                is_isolated_spike = (
                    abs(diff_left) > jump_threshold and
                    abs(diff_right) > jump_threshold and
                    diff_left * diff_right > 0
                )

                if is_isolated_spike:
                    denom = log_gamma[i + 1] - log_gamma[i - 1]
                    if abs(denom) > 1e-12:
                        w = (log_gamma[i] - log_gamma[i - 1]) / denom
                        res_fixed[i] = left + w * (right - left)
                    else:
                        res_fixed[i] = 0.5 * (left + right)

            df.loc[rows, "res_pred_fixed"] = res_fixed

        df = df.sort_values("_row_id")
        return df["res_pred_fixed"].to_numpy(dtype=float)

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Predict viscosity for new samples."""
        df = df_test.copy()
        df = df.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma"]).copy()

        df["Salt_raw"] = df["Salt"].astype(str)
        df_ml = self._preprocess_cat(df, is_training=False)

        eta_theory = self._theory_eta_batch(
            df[["Salt_raw", "Cs", "fs", "Cp", "T", "Gamma"]]
        )

        eta_pred = np.full(len(df_ml), np.nan, dtype=float)
        valid_mask = np.isfinite(eta_theory) & (eta_theory > 0)

        if not np.any(valid_mask):
            return eta_pred

        df_ml["Eta_Theory"] = eta_theory
        df_ml["Log_Eta_Theory"] = np.where(
            valid_mask,
            np.log10(df_ml["Eta_Theory"].values + 1e-12),
            np.nan
        )

        aT_global = self._get_aT(
            df_ml["T"].values.astype(float),
            self.rheo_E,
            self.rheo_Tref0
        )
        df_ml["aT_global"] = aT_global
        df_ml["Gamma_Reduced"] = (
            df_ml["Gamma"].astype(float).values *
            df_ml["aT_global"].astype(float).values
        )

        base_features = [
            "Salt", "Cs", "fs", "Cp",
            "Gamma", "aT_global", "Gamma_Reduced", "Log_Eta_Theory"
        ]
        if self.rheo_cfg.include_raw_T_feature:
            base_features = ["T"] + base_features

        X_valid = df_ml.loc[valid_mask, base_features]

        res_pred = self.rheo_ml.predict(X_valid)

        if bool(getattr(self.rheo_cfg, "use_residual_spike_repair", True)):
            if int(valid_mask.sum()) >= 3:
                df_info = df.loc[valid_mask, ["T", "Salt_raw", "Cs", "fs", "Cp", "Gamma"]].copy()
                res_pred = self._repair_isolated_residual_spikes_for_prediction(
                    df_info=df_info,
                    res_pred=res_pred,
                )

        y_pred_log = df_ml.loc[valid_mask, "Log_Eta_Theory"].values + res_pred
        eta_pred[valid_mask] = np.power(10.0, y_pred_log)
        return eta_pred

    def save(self, path: Path) -> None:
        """Save the full hybrid model to disk."""
        payload: Dict[str, Any] = {
            "rheo_cfg": self.rheo_cfg,
            "encoders": self.encoders,
            "rheo_cy_params": self.rheo_cy_params,
            "rheo_invalid_formulations": self.rheo_invalid_formulations,
            "rheo_E": self.rheo_E,
            "rheo_Tref0": self.rheo_Tref0,
            "rheo_ml": self.rheo_ml,
            "rheo_res_clip_low": self.rheo_res_clip_low,
            "rheo_res_clip_high": self.rheo_res_clip_high,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: Path, verbose: bool = True) -> "RheoHybridModel":
        """Load a saved hybrid model from disk."""
        payload = joblib.load(path)
        model = RheoHybridModel(payload["rheo_cfg"], verbose=verbose)

        model.encoders = payload["encoders"]
        model.rheo_cy_params = payload["rheo_cy_params"]
        model.rheo_invalid_formulations = payload.get("rheo_invalid_formulations", set())
        model.rheo_E = payload["rheo_E"]
        model.rheo_Tref0 = payload["rheo_Tref0"]
        model.rheo_ml = payload["rheo_ml"]
        model.rheo_res_clip_low = payload.get("rheo_res_clip_low", None)
        model.rheo_res_clip_high = payload.get("rheo_res_clip_high", None)
        return model



# ==============================================================================
# Main script
# ==============================================================================

if __name__ == "__main__":
    # Input file
    rheo_xlsx_path = DATA_DIR / "Rheo_Training_Data.xlsx"

    # Output files
    model_save_path = MODEL_DIR / "RheoHybridModel.joblib"
    processed_path = PROCESSED_DIR / "Rheo_Processed.csv"
    train_result_path = RESULT_DIR / "Rheo_Train_Result.csv"
    test_result_path = RESULT_DIR / "Rheo_Test_Result.csv"

    # Model configuration
    model = RheoHybridModel(
        rheo_cfg=RheoConfig(
            p_eta0_low=1.0,
            p_eta0_high=0.65,
            p_eta0_highT=100.0,
            fit_E_from_data=True,
            E_min=8000.0,
            E_max=2.0e5,
            use_temp_weights=True,
            weight_mode="inv_sqrt",
            max_weight_ratio=80.0,
            include_raw_T_feature=False,
            eta_min_keep=0.5,
            show_missing_formula_warning=True,
            plateau_gamma_abs=1e-2,
            plateau_bottomk=5,
            res_floor_apply_Tmin=100.0,
            res_floor_highT=-0.02,
            use_plateau_res_floor=True,
            use_residual_spike_repair=True,
            residual_spike_jump_threshold=0.30,
        ),
        verbose=True
    )

    try:
        print(">>> Reading Excel data ...")
        print(f">>> Input file: {rheo_xlsx_path}")

        # Training file columns: T, Salt, Cs, fs, Cp, Gamma, Eta
        df_rheo_raw = pd.read_excel(rheo_xlsx_path, header=1)
        df_rheo_raw.columns = ["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]

        # Step 1: preprocess and save the processed dataset
        df_rheo = model.preprocess_data(df_rheo_raw, save_path=processed_path)
        print(f">>> Number of total modeling samples: {len(df_rheo)}")

        # Step 2: fit physics parameters on the processed training dataset
        model.fit_physics(df_rheo)

        # Step 3: split the dataset into train and test sets by formulation group
        s_salt = df_rheo["Salt"].astype(str)
        s_Cs = df_rheo["Cs"].map(lambda x: f"{float(x):.8g}")
        s_fs = df_rheo["fs"].map(lambda x: f"{float(x):.8g}")
        s_Cp = df_rheo["Cp"].map(lambda x: f"{float(x):.8g}")
        rheo_groups = s_salt + "|" + s_Cs + "|" + s_fs + "|" + s_Cp

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        tr_idx, te_idx = next(gss.split(df_rheo, groups=rheo_groups))

        df_rheo_train = df_rheo.iloc[tr_idx].reset_index(drop=True)
        df_rheo_test = df_rheo.iloc[te_idx].reset_index(drop=True)

        # Step 4: train the residual model
        rheo_train_metrics = model.train(df_rheo_train.copy())

        # Training-set prediction
        rheo_pred_tr = model.predict(
            df_rheo_train[["T", "Salt", "Cs", "fs", "Cp", "Gamma"]].copy()
        )
        df_rheo_train_out = df_rheo_train.copy()
        df_rheo_train_out["Eta_pred"] = rheo_pred_tr
        df_rheo_train_out["Eta_abs_err"] = (
            df_rheo_train_out["Eta_pred"] - df_rheo_train_out["Eta"]
        ).abs()

        # Test-set prediction
        rheo_pred_te = model.predict(
            df_rheo_test[["T", "Salt", "Cs", "fs", "Cp", "Gamma"]].copy()
        )
        rheo_test_metrics = calc_eval_metrics(df_rheo_test["Eta"].values, rheo_pred_te)

        df_rheo_test_out = df_rheo_test.copy()
        df_rheo_test_out["Eta_pred"] = rheo_pred_te
        df_rheo_test_out["Eta_abs_err"] = (
            df_rheo_test_out["Eta_pred"] - df_rheo_test_out["Eta"]
        ).abs()

        # Print summary metrics
        print(
            f">>> Training results: "
            f"R2 = {rheo_train_metrics['R2']:.4f}, "
            f"RMSE = {rheo_train_metrics['RMSE']:.4e}, "
            f"MAE = {rheo_train_metrics['MAE']:.4e}, "
            f"Log_RMSE = {rheo_train_metrics['Log_RMSE']:.4f}"
        )
        print(
            f">>> Test results:     "
            f"R2 = {rheo_test_metrics['R2']:.4f}, "
            f"RMSE = {rheo_test_metrics['RMSE']:.4e}, "
            f"MAE = {rheo_test_metrics['MAE']:.4e}, "
            f"Log_RMSE = {rheo_test_metrics['Log_RMSE']:.4f}"
        )
        metrics_train_path = RESULT_DIR / "Rheo_Train_Metrics.csv"
        metrics_test_path = RESULT_DIR / "Rheo_Test_Metrics.csv"

        pd.DataFrame([rheo_train_metrics]).to_csv(metrics_train_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([rheo_test_metrics]).to_csv(metrics_test_path, index=False, encoding="utf-8-sig")

        # Save train/test result files
        df_rheo_train_out.to_csv(train_result_path, index=False, encoding="utf-8-sig")
        df_rheo_test_out.to_csv(test_result_path, index=False, encoding="utf-8-sig")

        print(f">>> Training results saved to: {train_result_path}")
        print(f">>> Test results saved to: {test_result_path}")

        # Step 5: save the trained model
        model.save(model_save_path)
        print(f">>> Model saved to: {model_save_path}")

        print("\n>>> Training completed.")
        print(">>> External prediction can be performed with predict_generalization_piml.py.")

    except Exception as e:
        import traceback
        print(f"\nRuntime error: {e}")
        traceback.print_exc()