# -*- coding: utf-8 -*-
"""
Predict Rheo_Generalization_Data.xlsx
using the trained RheoHybridModel.joblib
------------------------------------------------
This script is intentionally separate from training.

Workflow:
- Load the trained hybrid model
- Read the external generalization file
- Predict viscosity
- Save the prediction result file

"""

import __main__
from pathlib import Path

import pandas as pd

from train_piml import RheoConfig, RheoHybridModel

__main__.RheoConfig = RheoConfig
__main__.RheoHybridModel = RheoHybridModel


# ==============================================================================
# Paths
# ==============================================================================
CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent.parent
DATA_DIR = REPO_DIR / "data"

OUTPUT_DIR = CODE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"

# Ensure output folders exist
for d in [OUTPUT_DIR, MODEL_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Main script
# ==============================================================================

if __name__ == "__main__":
    # Input/output files
    model_path = MODEL_DIR / "RheoHybridModel.joblib"
    val_path = DATA_DIR / "Rheo_Generalization_Data.xlsx"
    out_path = RESULT_DIR / "Rheo_Generalization_Prediction.csv"

    print(">>> Loading model ...")
    print(f">>> Model file: {model_path}")

    # Load the trained model
    model = RheoHybridModel.load(model_path, verbose=True)

    print(">>> Reading generalization dataset ...")
    print(f">>> Input file: {val_path}")

    # Generalization file has 6 columns: T, Salt, Cs, fs, Cp, Gamma
    df_val = pd.read_excel(val_path, header=1)
    df_val.columns = ["T", "Salt", "Cs", "fs", "Cp", "Gamma"]

    # Basic cleaning before prediction
    df_val = df_val.dropna(subset=["T", "Salt", "Cs", "fs", "Cp", "Gamma"]).copy()
    df_val = df_val[df_val["Gamma"] > 0].copy()

    print(f">>> Number of generalization samples: {len(df_val)}")

    if len(df_val) == 0:
        raise ValueError("No valid samples were found in the generalization file.")

    print(">>> Running prediction ...")

    # Predict viscosity using the trained model
    eta_pred = model.predict(
        df_val[["T", "Salt", "Cs", "fs", "Cp", "Gamma"]].copy()
    )

    # Save row-wise prediction results
    df_out = df_val.copy()
    df_out["Eta_pred"] = eta_pred
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n>>> Prediction completed.")
    print(f">>> Prediction results saved to: {out_path}")