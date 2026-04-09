import numpy as np
import pandas as pd
import torch
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import DEVICE, setup_reproducibility
from src.data.loaders import load_abalone_data
from src.training.vaeac_trainer import train_vaeac
from src.analysis.shapley import estimate_shapley, vaeac_impute

def main():
    setup_reproducibility()
    print(f"Device: {DEVICE}")

    # 1. Load Data
    print("Loading Abalone dataset...")
    df, X_scaled, y = load_abalone_data()
    print(f"Data shape: {X_scaled.shape}")

    # 2. Simulate Missing Data
    print("Simulating 20% missing data...")
    mask_missing = np.random.rand(*X_scaled.shape) < 0.2
    X_miss = X_scaled.copy()
    X_miss[mask_missing] = np.nan

    # 3. Train VAEAC
    print("Training VAEAC model (this may take a while)...")
    vaeac_model = train_vaeac(X_miss, imputation_method='mean', epochs=20, batch_size=64)

    # 4. Impute Data for Downstream Task Training
    print("Imputing data for Random Forest training...")
    # For simplicity in this demo, we use a simple imputation for RF training
    from sklearn.impute import SimpleImputer
    X_clean_rf = SimpleImputer().fit_transform(X_miss)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_clean_rf, y)
    print("Random Forest trained.")

    # 5. Explainability with Shapley
    idx = 10
    x_test = X_scaled[idx]
    print(f"Calculating Shapley values for instance {idx}...")

    shap_vals = estimate_shapley(vaeac_model, rf_model, x_test, n_coalitions=50, n_samples_mc=20)

    print("\nShapley Values:")
    feature_names = df.columns[:-1]
    for name, val in zip(feature_names, shap_vals):
        print(f"{name}: {val:.4f}")

if __name__ == "__main__":
    main()
