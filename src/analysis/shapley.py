import torch
import numpy as np
from src.config import DEVICE

def vaeac_impute(model, x_batch, mask, K=10):
    """Génère K imputations via le Masked Encoder + Decoder (Inférence)."""
    model.eval()
    with torch.no_grad():
        x_batch = x_batch.to(DEVICE)
        mask = mask.to(DEVICE)

        # x_masked : zéros là où mask=1
        x_masked_input = x_batch.clone()
        x_masked_input[mask.bool()] = 0

        # Forward Masked Encoder
        mu, logvar, skips = model.forward_masked_encoder(x_masked_input, mask)

        imputations = []
        for k in range(K):
            z = model.reparameterize(mu, logvar)
            rec_mu, _ = model.forward_decoder(z, skips)

            # Combiner: Valeurs observées + Prédictions
            x_final = x_batch * (1 - mask) + rec_mu * mask
            imputations.append(x_final.cpu().numpy())

        return np.array(imputations) # [K, Batch, Features]

def estimate_shapley(vaeac_model, predictor, x_instance, n_coalitions=100, n_samples_mc=10):
    """Estimation Monte Carlo des valeurs de Shapley."""
    M = len(x_instance)
    phi = np.zeros(M)
    x_tensor = torch.FloatTensor(x_instance).unsqueeze(0) # [1, M]

    # Baseline value (E[f(x)])
    mask_all = torch.ones_like(x_tensor)
    imps_all = vaeac_impute(vaeac_model, x_tensor, mask_all, K=n_samples_mc)
    preds_base = [predictor.predict(imps_all[k])[0] for k in range(n_samples_mc)]
    base_value = np.mean(preds_base)

    # Boucle sur coalitions
    for _ in range(n_coalitions):
        perm = np.random.permutation(M)
        x_S = x_tensor.clone()
        mask = torch.ones_like(x_tensor) # Tout masqué
        prev_val = base_value

        for feature_idx in perm:
            # Ajouter feature i à la coalition S
            mask[0, feature_idx] = 0

            # Calculer E[f(x) | S U {i}] via VAEAC
            imps = vaeac_impute(vaeac_model, x_S, mask, K=n_samples_mc)
            preds = [predictor.predict(imps[k])[0] for k in range(n_samples_mc)]
            curr_val = np.mean(preds)

            # Contribution marginale
            phi[feature_idx] += (curr_val - prev_val)
            prev_val = curr_val

    return phi / n_coalitions

def estimate_shapley_generic(impute_fn, predictor, x_instance, M, n_coalitions=50, n_samples_mc=10):
    """Version générique de l'estimateur Shapley acceptant n'importe quelle fonction d'imputation."""
    phi = np.zeros(M)
    x_tensor = torch.FloatTensor(x_instance).unsqueeze(0).to(DEVICE)

    # Baseline
    mask_all = torch.ones_like(x_tensor)
    imps_all = impute_fn(x_tensor, mask_all, K=n_samples_mc)
    # Moyenne des prédictions sur imputations
    preds_base = [predictor.predict(imps_all[k])[0] for k in range(n_samples_mc)]
    base_value = np.mean(preds_base)

    for _ in range(n_coalitions):
        perm = np.random.permutation(M)
        x_S = x_tensor.clone() # Départ: tout masqué (ou presque, selon logique)
        # Logique standard: On part de Vide, on ajoute 1 par 1
        # Masque: 1=Manquant. Au début tout est manquant (1).
        mask = torch.ones_like(x_tensor)

        prev_val = base_value

        for feature_idx in perm:
            # On observe feature_idx (Masque passe à 0)
            mask[0, feature_idx] = 0

            # Imputation conditionnelle P(X_missing | X_observed)
            imps = impute_fn(x_S, mask, K=n_samples_mc)

            # Prédiction moyenne
            preds = [predictor.predict(imps[k])[0] for k in range(n_samples_mc)]
            curr_val = np.mean(preds)

            phi[feature_idx] += (curr_val - prev_val)
            prev_val = curr_val

    return phi / n_coalitions

def predict_shapley_projected(dln_model, rf_model, x_instance, base_value):
    """
    Predit les valeurs de Shapley avec garantie mathématique d'Efficacité.
    Phi_proj = Phi_raw + (Erreur / M)
    """
    # 1. Prédiction brute du DLN (Rapide)
    dln_model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_instance).unsqueeze(0).to(DEVICE)
        phi_raw = dln_model(x_tensor).cpu().numpy()[0]

    # 2. Calcul du gap (Erreur résiduelle vis-à-vis de la prédiction RF)
    # Cible réelle : f(x) - E[f(x)]
    pred_fx = rf_model.predict(x_instance.reshape(1, -1))[0]
    target_sum = pred_fx - base_value

    # Somme actuelle prédite par le réseau
    current_sum = np.sum(phi_raw)
    residual = target_sum - current_sum

    # 3. Projection (Répartition uniforme de l'erreur résiduelle)
    M = len(phi_raw)
    phi_projected = phi_raw + (residual / M)

    return phi_projected
