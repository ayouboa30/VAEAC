import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

from src.config import DEVICE
from src.training.vaeac_trainer import train_vaeac, train_vaeac_synthetic
from src.analysis.shapley import vaeac_impute, estimate_shapley_generic
from src.utils.metrics import calculate_kl_divergence
from src.data.loaders import generate_data

class TrueGaussianImputer:

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def impute(self, x_batch_tensor, mask_tensor, K=10):
        """
        x_batch: [N, D] (avec zéros ou nans)
        mask: [N, D] (0=Observé, 1=Manquant)
        """
        N, D = x_batch_tensor.shape
        x_np = x_batch_tensor.cpu().numpy()
        m_np = mask_tensor.cpu().numpy().astype(bool)

        imputations = np.zeros((K, N, D))

        for i in range(N):
            # Indices
            missing_idx = np.where(m_np[i])[0]
            observed_idx = np.where(~m_np[i])[0]

            if len(missing_idx) == 0:
                imputations[:, i, :] = x_np[i]
                continue

            # Partitionnement Mean / Covariance
            mu_1 = self.mean[missing_idx]
            mu_2 = self.mean[observed_idx]

            Sigma_11 = self.cov[np.ix_(missing_idx, missing_idx)]
            Sigma_12 = self.cov[np.ix_(missing_idx, observed_idx)]
            Sigma_21 = self.cov[np.ix_(observed_idx, missing_idx)]
            Sigma_22 = self.cov[np.ix_(observed_idx, observed_idx)]

            # Formule conditionnelle Gaussienne
            # mu_cond = mu_1 + S12 * inv(S22) * (x_obs - mu_2)
            x_obs = x_np[i, observed_idx]

            # Gestion numérique de l'inverse (pseudo-inverse pour stabilité)
            if len(observed_idx) > 0:
                cond_mean = mu_1 + Sigma_12 @ np.linalg.pinv(Sigma_22) @ (x_obs - mu_2)
                cond_cov = Sigma_11 - Sigma_12 @ np.linalg.pinv(Sigma_22) @ Sigma_21
            else:
                cond_mean = mu_1
                cond_cov = Sigma_11

            # Sampling K fois
            samples = np.random.multivariate_normal(cond_mean, cond_cov, size=K)

            # Remplissage
            imputations[:, i, :] = x_np[i] # Copie base
            imputations[:, i, missing_idx] = samples

        return imputations

def analyze_shapley_fidelity(dimensions=[5, 10, 20], n_train=1000):
    shap_results = []

    # Définition des Covariances Théoriques
    def get_configs(dim):
        # 1. Indépendant: Identité
        cov_indep = np.eye(dim)

        # 2. Markov (AR1): Sigma_ij = rho^|i-j|
        rho = 0.8
        indices = np.arange(dim)
        cov_markov = rho ** np.abs(indices[:, None] - indices[None, :])

        # 3. Copule (Equicorrélée): Tous corrélation 0.5
        cov_copula = np.full((dim, dim), 0.5)
        np.fill_diagonal(cov_copula, 1.0)

        return {
            "Indépendant": cov_indep,
            "Markov Chain": cov_markov,
            "Copule Gaussienne": cov_copula
        }

    for dim in dimensions:
        print(f"\n>>> Shapley Fidelity | Dimension M = {dim}")
        configs = get_configs(dim)

        for dtype_name, true_cov in configs.items():
            print(f"   Dataset: {dtype_name}...")

            # A. Génération Données
            true_mean = np.zeros(dim)
            X_gen = np.random.multivariate_normal(true_mean, true_cov, size=n_train).astype(np.float32)
            # Cible synthétique non-linéaire (pour justifier Shapley)
            # y = sum(x) + interaction(x0, x1)
            y_gen = np.sum(X_gen, axis=1) + 2 * X_gen[:, 0] * X_gen[:, 1]

            # B. Entraînement Modèle Black-Box (RF)
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(X_gen, y_gen)

            # C. Entraînement VAEAC (sur données avec trous simulés)
            mask_train = np.random.rand(*X_gen.shape) < 0.2
            X_miss_train = X_gen.copy()
            X_miss_train[mask_train] = np.nan
            vaeac_model = train_vaeac(X_miss_train, imputation_method='mean', epochs=10, batch_size=64)

            # D. Comparaison Shapley (sur 5 instances de test)
            n_test_instances = 5
            mae_list = []

            # Wrappers pour les fonctions d'imputation
            true_imputer = TrueGaussianImputer(true_mean, true_cov)

            def wrap_vaeac(x, m, K):
                return vaeac_impute(vaeac_model, x, m, K)

            def wrap_true(x, m, K):
                # Conversion torch -> numpy interne à la classe, mais on garde signature
                return true_imputer.impute(x, m, K)

            for i in range(n_test_instances):
                x_test = X_gen[i] # Instance complète (Shapley masque lui-même)

                # 1. Vrai Shapley
                phi_true = estimate_shapley_generic(wrap_true, rf, x_test, dim, n_coalitions=40, n_samples_mc=10)

                # 2. VAEAC Shapley
                phi_vaeac = estimate_shapley_generic(wrap_vaeac, rf, x_test, dim, n_coalitions=40, n_samples_mc=10)

                # Erreur
                mae = np.mean(np.abs(phi_true - phi_vaeac))
                mae_list.append(mae)

            avg_mae = np.mean(mae_list)
            shap_results.append({
                "Dimension": dim,
                "Dépendance": dtype_name,
                "Shapley MAE": avg_mae
            })
            print(f"   -> MAE Erreur: {avg_mae:.4f}")

    return pd.DataFrame(shap_results)

def run_sparsity_test():
    sparsity_levels = [0.0, 0.2, 0.4, 0.6, 0.8] # % de la matrice qui est VRAIMENT zéro
    fixed_missing_rate = 0.3 # On cache 30% des données

    rmse_by_sparsity = {'Mean': [], 'Zero': [], 'Iterative': [], 'KNN': [], 'VAEAC': []}

    print("\n>>> Démarrage du Test de Sparsité (Influence des Zéros) <<<")

    for sp in sparsity_levels:
        print(f"  -> Test avec {sp*100:.0f}% de zéros dans la vérité terrain...")
        # On génère des données IID mais avec beaucoup de zéros
        X_sparse = generate_data(data_type='iid', sparsity=sp)

        # On crée les trous (NaN)
        mask_miss = np.random.rand(*X_sparse.shape) < fixed_missing_rate
        X_miss = X_sparse.copy()
        X_miss[mask_miss] = np.nan

        # --- Modèles ---
        methods = {
            'Mean': SimpleImputer(strategy='mean'),
            'Zero': None, # Impute par 0
            'Iterative': IterativeImputer(max_iter=10),
            'KNN': KNNImputer(n_neighbors=5)
        }

        for name, model in methods.items():
            if name == 'Zero':
                X_imp = np.nan_to_num(X_miss, nan=0.0)
            else:
                X_imp = model.fit_transform(X_miss)

            rmse = np.sqrt(np.mean((X_sparse[mask_miss] - X_imp[mask_miss])**2))
            rmse_by_sparsity[name].append(rmse)

        # VAEAC
        try:
            model_synth = train_vaeac_synthetic(X_miss, epochs=20) # Moins d'epochs pour aller vite
            X_in_tens = torch.FloatTensor(np.nan_to_num(X_miss, nan=0.0)).to(DEVICE)
            mask_tens = torch.FloatTensor(np.isnan(X_miss)).to(DEVICE)
            model_synth.eval()
            with torch.no_grad():
                mu, logvar, skips = model_synth.forward_masked_encoder(X_in_tens, mask_tens)
                z = model_synth.reparameterize(mu, logvar)
                rec, _ = model_synth.forward_decoder(z, skips)
                X_rec = rec.cpu().numpy()

            X_vaeac = X_miss.copy()
            X_vaeac[mask_miss] = X_rec[mask_miss]
            rmse_vaeac = np.sqrt(np.mean((X_sparse[mask_miss] - X_vaeac[mask_miss])**2))
            rmse_by_sparsity['VAEAC'].append(rmse_vaeac)
        except:
             rmse_by_sparsity['VAEAC'].append(0)

    # --- Plot Sparsity ---
    plt.figure(figsize=(10, 6))
    for name, scores in rmse_by_sparsity.items():
        plt.plot(sparsity_levels, scores, marker='o', label=name, linewidth=2)

    plt.title(f"Performance vs Sparsité de la Matrice (NaN fixed @ {fixed_missing_rate*100:.0f}%)")
    plt.xlabel("Pourcentage de Zéros dans la Vérité Terrain (Sparsity)")
    plt.ylabel("RMSE (Erreur)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sparsity_test.png')
    plt.show()

def run_imputation_experiment(X_ground_truth, title="Analyse"):
    """
    Exécute la comparaison des méthodes sur un dataset donné.
    """
    missing_rates = [0.1, 0.3, 0.5, 0.7] # Taux de valeurs manquantes (NaN)

    results_rmse = {'Mean': [], 'Zero': [], 'Iterative': [], 'KNN': [], 'VAEAC': []}
    results_kl = {'Mean': [], 'Zero': [], 'Iterative': [], 'KNN': [], 'VAEAC': []}
    saved_imputations = {} # Pour le plot de densité à 50%

    print(f"\n>>> Traitement : {title} <<<")

    for rate in missing_rates:
        # Masque de valeurs manquantes (NaN)
        mask_miss = np.random.rand(*X_ground_truth.shape) < rate
        X_miss = X_ground_truth.copy()
        X_miss[mask_miss] = np.nan

        # --- Méthodes Classiques ---
        methods = {
            'Mean': SimpleImputer(strategy='mean'),
            'Zero': None,
            'Iterative': IterativeImputer(max_iter=10, random_state=42),
            'KNN': KNNImputer(n_neighbors=5)
        }

        X_imputed_current = {}

        for name, model in methods.items():
            if name == 'Zero':
                X_imp = np.nan_to_num(X_miss, nan=0.0)
            else:
                X_imp = model.fit_transform(X_miss)

            X_imputed_current[name] = X_imp

            # Calcul Métriques
            # RMSE calculé uniquement sur les parties manquantes
            if np.sum(mask_miss) > 0:
                mse = np.mean((X_ground_truth[mask_miss] - X_imp[mask_miss])**2)
                rmse = np.sqrt(mse)
            else:
                rmse = 0.0

            kl = calculate_kl_divergence(X_ground_truth, X_imp)

            results_rmse[name].append(rmse)
            results_kl[name].append(kl)

        # --- VAEAC (Intégration) ---
        try:
            # Note: Assurez-vous que train_vaeac_synthetic est bien défini plus haut
            model_synth = train_vaeac_synthetic(X_miss, epochs=30)

            X_in_tens = torch.FloatTensor(np.nan_to_num(X_miss, nan=0.0)).to(DEVICE)
            mask_tens = torch.FloatTensor(np.isnan(X_miss)).to(DEVICE)

            model_synth.eval()
            with torch.no_grad():
                mu, logvar, skips = model_synth.forward_masked_encoder(X_in_tens, mask_tens)
                preds = []
                for _ in range(5): # 5 passes MCMC pour lisser
                    z = model_synth.reparameterize(mu, logvar)
                    rec, _ = model_synth.forward_decoder(z, skips)
                    preds.append(rec.cpu().numpy())
                X_rec = np.mean(preds, axis=0)

            X_vaeac = X_miss.copy()
            X_vaeac[mask_miss] = X_rec[mask_miss]

            if np.sum(mask_miss) > 0:
                rmse_vaeac = np.sqrt(np.mean((X_ground_truth[mask_miss] - X_vaeac[mask_miss])**2))
            else:
                rmse_vaeac = 0.0

            kl_vaeac = calculate_kl_divergence(X_ground_truth, X_vaeac)

            results_rmse['VAEAC'].append(rmse_vaeac)
            results_kl['VAEAC'].append(kl_vaeac)
            X_imputed_current['VAEAC'] = X_vaeac

        except NameError:
            print("Attention: Fonction train_vaeac_synthetic non trouvée.")
            results_rmse['VAEAC'].append(0)
            results_kl['VAEAC'].append(0)
        except Exception as e:
            print(f"Erreur VAEAC: {e}")
            results_rmse['VAEAC'].append(0)
            results_kl['VAEAC'].append(0)

        if rate == 0.5:
            saved_imputations = X_imputed_current

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Résultats pour : {title}", fontsize=16)

    # RMSE
    for name, scores in results_rmse.items():
        if len(scores) > 0: axes[0].plot(missing_rates, scores, marker='o', label=name)
    axes[0].set_title("RMSE (Reconstruction)")
    axes[0].set_xlabel("% Manquant (NaN)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KL
    for name, scores in results_kl.items():
         if len(scores) > 0: axes[1].plot(missing_rates, scores, marker='s', label=name)
    axes[1].set_title("Divergence KL (Distribution)")
    axes[1].set_xlabel("% Manquant (NaN)")
    axes[1].set_ylabel("KL")
    axes[1].grid(True, alpha=0.3)

    # Densité
    if len(saved_imputations) > 0:
        feature_idx = 0
        sns.kdeplot(X_ground_truth[:, feature_idx], ax=axes[2], color='black', fill=True, alpha=0.1, label='Vrai', linewidth=3)
        colors = {'Mean': 'red', 'Zero': 'purple', 'Iterative': 'green', 'KNN': 'orange', 'VAEAC': 'blue'}
        for name, X_imp in saved_imputations.items():
            sns.kdeplot(X_imp[:, feature_idx], ax=axes[2], color=colors.get(name, 'gray'), label=name, linestyle='--')
        axes[2].set_title("Densité (Taux Manquant=50%)")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_results.png')
    plt.show()

def prepare_distillation_dataset(X, teacher_model, n_clusters=50, k_neighbors=5):
    """
    Crée un dataset d'entraînement réduit combinant :
    1. La structure globale (Centroïdes K-Means)
    2. La variance locale (k-Nearest Neighbors réels)
    """
    print(f"🔄 Stratégie d'échantillonnage : {n_clusters} Barycentres + {k_neighbors} Voisins/cluster...")

    # 1. K-Means pour trouver les barycentres
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    # 2. k-NN pour trouver les points réels autour des barycentres
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(centroids)

    # Récupération des voisins (aplatissement des indices)
    neighbor_indices = indices.flatten()
    X_neighbors = X[neighbor_indices]

    # 3. Fusion (Barycentres + Voisins) et suppression des doublons
    X_distill = np.vstack([centroids, X_neighbors])
    X_distill = np.unique(X_distill, axis=0)

    print(f"✅ Dataset de Distillation créé : {X_distill.shape[0]} échantillons")
    print(f"   (vs {X.shape[0]} dans le dataset complet)")

    # 4. Labellisation (Teacher) - C'est l'étape coûteuse, mais sur peu de points
    print("🎯 Calcul des cibles Shapley (Teacher) pour le subset réduit...")
    explainer = shap.TreeExplainer(teacher_model)
    y_distill = explainer.shap_values(X_distill)

    return X_distill, y_distill
