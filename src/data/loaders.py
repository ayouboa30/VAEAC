import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_abalone_data():
    # Chargement du dataset Abalone
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    cols = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]
    df = pd.read_csv(url, names=cols)

    # Encodage de la variable catégorielle 'Sex' (F, M, I -> 0, 1, 2)
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

    # Séparation Features (X) et Cible (y)
    X = df.drop(columns=["Rings"]).values.astype(np.float32)
    y = df["Rings"].values.astype(np.float32)

    # Normalisation (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y

def generate_data(data_type='iid', n_samples=2000, n_features=5, sparsity=0.0):
    """
    Génère des données synthétiques avec option de sparsité (zéros réels).
    sparsity: Pourcentage de valeurs qui seront mises à 0.0 (simule des features inutiles/nulles)
    """
    np.random.seed(42)

    if data_type == 'iid':
        X = np.random.randn(n_samples, n_features)

    elif data_type == 'mixture':
        n1 = n_samples // 2
        n2 = n_samples - n1
        x1 = np.random.randn(n1, n_features) - 2.5
        x2 = np.random.randn(n2, n_features) + 2.5
        X = np.vstack([x1, x2])

    elif data_type == 'copula':
        # Copule simplifiée pour l'exemple
        cov = np.ones((n_features, n_features)) * 0.85
        np.fill_diagonal(cov, 1.0)
        z = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)
        from scipy.stats import norm, expon
        X = expon.ppf(norm.cdf(z))

    else:
        raise ValueError("Type inconnu")

    # --- AJOUT DE LA SPARSITÉ (ZÉROS RÉELS) ---
    if sparsity > 0:
        # On force aléatoirement des valeurs à 0
        mask_zeros = np.random.rand(*X.shape) < sparsity
        X[mask_zeros] = 0.0

    return X
