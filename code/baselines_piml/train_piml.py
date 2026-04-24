# -*- coding: utf-8 -*-
"""
Physics-Informed Machine Learning (PIML) for Rheology Modeling
----------------------------------------------------------------
Main features:
1) Introduce temperature through the Arrhenius shift factor aT
   (definition: at lower temperature, aT > 1)
2) Use the Carreau–Yasuda (CY) model as the physics-based baseline:
   eta_theory(gamma, T)
3) Train a random forest on log-space residuals:
   residual = log10(eta) - log10(eta_theory)
4) Support temperature-imbalance sample weighting with capped weights
5) Preserve segmented high-temperature p_eta0 for extrapolation
6) Use the full training dataset for preprocessing, physics fitting, and train/test split
7) External generalization prediction is handled separately by predict_generalization_piml.py

Workflow:
- Read raw rheology training data from Excel
- Preprocess and aggregate repeated measurements
- Fit physics parameters on the processed training data
- Split processed data into training and test sets
- Train the random-forest residual model
- Save train/test predictions and the trained model

Important:
This training script does NOT run external generalization prediction.
To predict the generalization samples, run predict_generalization_piml.py separately.
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
from sklearn.metrics import mean_squared_error, r2_score

import joblib


# ==============================================================================
# Paths
# ==============================================================================
CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent.parent
DATA_DIR = REPO_DIR / "data"
# All generated files are stored under code/baselines_piml/outputs/
OUTPUT_DIR = CODE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PROCESSED_DIR = OUTPUT_DIR / "processed"
RESULT_DIR = OUTPUT_DIR / "results"

# Create output folders automatically if they do not exist
for d in [OUTPUT_DIR, MODEL_DIR, PROCESSED_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Physics formulas
# ==============================================================================

class PhysicsFormulas:
    """Collection of rheology-related physics formulas."""
    R = 8.314  # Gas constant, J/mol/K

    @staticmethod
    def arrhenius_log10_aT(T_K: np.ndarray, E: float, Tref_K: float) -> np.ndarray:
        """
        Arrhenius shift factor in log10 form.
        Definition:
            at lower temperature, aT > 1
        Formula:
            log10(aT) = +(E / (R * ln(10))) * (1/T - 1/Tref)
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
        Carreau–Yasuda viscosity model.
        Formula:
            eta(gamma) = eta_inf + (eta0 - eta_inf) *
                         (1 + (lam * gamma)^a)^((n - 1) / a)
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
    # Global reference temperature in Celsius
    Tref_c: float = 25.0

    # Segmented exponent for eta0(T) at high temperature
    p_eta0_low: float = 1.0
    p_eta0_high: float = 0.75
    p_eta0_highT: float = 100.0

    # Default activation energy if fitting from data fails
    default_E: float = 5.0e4  # J/mol

    # Whether to fit a global activation energy from data
    fit_E_from_data: bool = True

    # Allowed range for fitted activation energy
    E_min: float = 8.0e3
    E_max: float = 2.0e5

    # Whether to apply sample weighting to reduce temperature imbalance
    use_temp_weights: bool = True

    # Weighting mode
    weight_mode: str = "inv_sqrt"
    max_weight_ratio: float = 80.0

    # Whether to include raw T as an ML feature
    include_raw_T_feature: bool = False

    # Remove very small viscosity values before fitting/training
    eta_min_keep: float = 0.5

    # Show warning if a formulation is missing fitted CY parameters
    show_missing_formula_warning: bool = True

    # Plateau-region identification parameters
    plateau_gamma_abs: float = 1e-2
    plateau_bottomk: int = 5

    # Apply a lower bound to residuals in the high-temperature plateau region
    res_floor_apply_Tmin: float = 100.0
    res_floor_highT: float = -0.02
    use_plateau_res_floor: bool = True


# ==============================================================================
# Main hybrid model
# ==============================================================================

class RheoHybridModel:
    """
    Hybrid rheology model:
    - Physics part: Carreau–Yasuda + Arrhenius temperature shift
    - ML part: random forest trained on residuals in log space
    """

    def __init__(self,
                 rheo_cfg: RheoConfig = RheoConfig(),
                 verbose: bool = True):
        self.rheo_cfg = rheo_cfg
        self.verbose = verbose

        # Encoder for categorical variable Salt
        self.encoders: Dict[str, LabelEncoder] = {}

        # Fitted CY parameters for each formulation
        self.rheo_cy_params: Dict[Tuple[str, float, float, float], Dict[str, float]] = {}

        # Global Arrhenius parameters
        self.rheo_E: float = rheo_cfg.default_E
        self.rheo_Tref0: float = rheo_cfg.Tref_c + 273.15  # Kelvin

        # Random-forest model for residual learning
        self.rheo_ml = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=0
        )

    def _log(self, msg: str) -> None:
        """Print log message when verbose mode is on."""
        if self.verbose:
            print(msg)

    def _preprocess_cat(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Encode the categorical variable Salt."""
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
        """Convert Celsius to Kelvin and compute Arrhenius shift factor."""
        T_K = np.asarray(T_c, dtype=float) + 273.15
        log10_aT = PhysicsFormulas.arrhenius_log10_aT(T_K, E, Tref_K)
        return np.power(10.0, log10_aT)

    def fit_physics(self, df: pd.DataFrame) -> None:
        """
        Fit the physics part of the hybrid model:
        1) Carreau–Yasuda parameters for each formulation
        2) A global activation energy E (optional)
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
        """Fit Carreau–Yasuda parameters for each formulation."""
        self.rheo_cy_params = {}

        for (salt, Cs, fs, Cp), g in df.groupby(["Salt", "Cs", "fs", "Cp"]):
            temps = np.array(sorted(g["T"].unique()))
            if len(temps) == 0:
                continue

            Tref_used = 25.0 if 25.0 in temps else float(temps[np.argmin(np.abs(temps - 25.0))])
            Tref_used_K = Tref_used + 273.15
            gref = g[g["T"] == Tref_used].copy()

            if len(gref) < 6:
                if len(gref) > 0:
                    y = gref["Eta"].values.astype(float)
                    eta0_ref = float(np.max(y))
                    eta_inf = float(max(np.min(y) * 0.1, 0.0))
                else:
                    eta0_ref, eta_inf = 1.0, 0.0

                self.rheo_cy_params[(str(salt), float(Cs), float(fs), float(Cp))] = {
                    "Tref_K": Tref_used_K,
                    "eta0_ref": eta0_ref,
                    "eta_inf": eta_inf,
                    "lam_ref": 1.0,
                    "n": 0.5,
                    "a": 2.0,
                }
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
                eta0_ref, eta_inf, lam_ref, n, a = [float(v) for v in p0]

            self.rheo_cy_params[(str(salt), float(Cs), float(fs), float(Cp))] = {
                "Tref_K": Tref_used_K,
                "eta0_ref": eta0_ref,
                "eta_inf": eta_inf,
                "lam_ref": lam_ref,
                "n": n,
                "a": a,
            }

    def _fit_E_from_data(self, df: pd.DataFrame) -> Optional[float]:
        """Estimate a global Arrhenius activation energy E from data."""
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
        """Compute the physics-based baseline viscosity for a batch of samples."""
        eta_out = np.empty(len(df), dtype=float)
        df_local = df.reset_index(drop=True)

        for (salt, Cs, fs, Cp), idx in df_local.groupby(["Salt_raw", "Cs", "fs", "Cp"]).groups.items():
            rows = np.array(list(idx), dtype=int)
            sub = df_local.loc[rows]

            key = (str(salt), float(Cs), float(fs), float(Cp))
            p = self.rheo_cy_params.get(key, None)

            if p is None:
                if self.rheo_cfg.show_missing_formula_warning and self.verbose:
                    print(
                        "[WARN] Missing CY formulation parameters: "
                        f"Salt={salt}, Cs={Cs}, fs={fs}, Cp={Cp}"
                    )

                eta0_ref, eta_inf, lam_ref, n, a = 1.0, 0.0, 1.0, 0.5, 2.0
                Tref_K = self.rheo_Tref0
            else:
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
        """Compute sample weights to reduce temperature imbalance."""
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
        """Identify plateau-region points."""
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
        Preprocess raw rheology data:
        - remove rows with very small viscosity
        - aggregate repeated measurements by median Eta
        - save the processed dataset
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
        """Train the random-forest residual model on the training set."""
        self._log(">>> [Rheo] Training ...")

        df_train = df_train.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]).copy()
        df_train = df_train[df_train["Eta"] > float(self.rheo_cfg.eta_min_keep)].copy()

        df_feat = df_train.copy()
        df_feat["Salt_raw"] = df_feat["Salt"].astype(str)

        df_ml = self._preprocess_cat(df_feat, is_training=True)

        eta_theory = self._theory_eta_batch(
            df_feat[["Salt_raw", "Cs", "fs", "Cp", "T", "Gamma"]]
        )
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

        mse = mean_squared_error(df_ml["Eta"].values, y_pred)
        r2 = r2_score(df_ml["Eta"].values, y_pred)
        return float(mse), float(r2)

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Predict viscosity for new samples."""
        df = df_test.copy()
        df = df.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma"]).copy()

        df["Salt_raw"] = df["Salt"].astype(str)
        df_ml = self._preprocess_cat(df, is_training=False)

        eta_theory = self._theory_eta_batch(
            df[["Salt_raw", "Cs", "fs", "Cp", "T", "Gamma"]]
        )
        df_ml["Eta_Theory"] = eta_theory
        df_ml["Log_Eta_Theory"] = np.log10(df_ml["Eta_Theory"].values + 1e-12)

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

        res_pred = self.rheo_ml.predict(X)
        y_pred_log = df_ml["Log_Eta_Theory"].values + res_pred

        eta_pred = np.power(10.0, y_pred_log)
        return eta_pred

    def save(self, path: Path) -> None:
        """Save the full hybrid model to disk."""
        payload: Dict[str, Any] = {
            "rheo_cfg": self.rheo_cfg,
            "encoders": self.encoders,
            "rheo_cy_params": self.rheo_cy_params,
            "rheo_E": self.rheo_E,
            "rheo_Tref0": self.rheo_Tref0,
            "rheo_ml": self.rheo_ml,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: Path, verbose: bool = True) -> "RheoHybridModel":
        """Load a saved hybrid model from disk."""
        payload = joblib.load(path)
        model = RheoHybridModel(payload["rheo_cfg"], verbose=verbose)

        model.encoders = payload["encoders"]
        model.rheo_cy_params = payload["rheo_cy_params"]
        model.rheo_E = payload["rheo_E"]
        model.rheo_Tref0 = payload["rheo_Tref0"]
        model.rheo_ml = payload["rheo_ml"]
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

    # Initialize the model with user-defined settings
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
        ),
        verbose=True
    )

    try:
        print(">>> Reading Excel data ...")
        print(f">>> Input file: {rheo_xlsx_path}")

        # Training file has 7 columns: T, Salt, Cs, fs, Cp, Gamma, Eta
        df_rheo_raw = pd.read_excel(rheo_xlsx_path, header=1)
        df_rheo_raw.columns = ["T", "Salt", "Cs", "fs", "Cp", "Gamma", "Eta"]

        # Step 1: preprocess and save processed data
        df_rheo = model.preprocess_data(df_rheo_raw, save_path=processed_path)
        print(f">>> Number of total modeling samples: {len(df_rheo)}")

        # Step 2: fit physics parameters on all processed training data
        model.fit_physics(df_rheo)

        # Step 3: split data into train/test by formulation group
        s_salt = df_rheo["Salt"].astype(str)
        s_Cs = df_rheo["Cs"].map(lambda x: f"{float(x):.8g}")
        s_fs = df_rheo["fs"].map(lambda x: f"{float(x):.8g}")
        s_Cp = df_rheo["Cp"].map(lambda x: f"{float(x):.8g}")
        rheo_groups = s_salt + "|" + s_Cs + "|" + s_fs + "|" + s_Cp

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        tr_idx, te_idx = next(gss.split(df_rheo, groups=rheo_groups))

        df_rheo_train = df_rheo.iloc[tr_idx].reset_index(drop=True)
        df_rheo_test = df_rheo.iloc[te_idx].reset_index(drop=True)

        # Step 4: train the ML residual model
        rheo_mse_tr, rheo_r2_tr = model.train(df_rheo_train.copy())

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
        rheo_mse_te = mean_squared_error(df_rheo_test["Eta"].values, rheo_pred_te)
        rheo_r2_te = r2_score(df_rheo_test["Eta"].values, rheo_pred_te)

        df_rheo_test_out = df_rheo_test.copy()
        df_rheo_test_out["Eta_pred"] = rheo_pred_te
        df_rheo_test_out["Eta_abs_err"] = (
            df_rheo_test_out["Eta_pred"] - df_rheo_test_out["Eta"]
        ).abs()

        # Print summary metrics
        print(f">>> Training results: MSE = {rheo_mse_tr:.4e}, R2 = {rheo_r2_tr:.4f}")
        print(f">>> Test results:     MSE = {rheo_mse_te:.4e}, R2 = {rheo_r2_te:.4f}")

        # Save train/test result files
        df_rheo_train_out.to_csv(train_result_path, index=False, encoding="utf-8-sig")
        df_rheo_test_out.to_csv(test_result_path, index=False, encoding="utf-8-sig")

        print(f">>> Training results saved to: {train_result_path}")
        print(f">>> Test results saved to: {test_result_path}")

        # Step 5: save the trained model
        model.save(model_save_path)
        print(f">>> Model saved to: {model_save_path}")

        print("\n>>> Training completed.")
        print(">>> Run predict_generalization_piml.py separately for external prediction.")

    except Exception as e:
        import traceback
        print(f"\nRuntime error: {e}")
        traceback.print_exc()