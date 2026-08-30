import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from src.config import DEVICE
from src.models.vaeac import VAEAC_Network
from src.data.mask_generator import MaskGenerator
from src.utils.losses import vaeac_loss, vaeac_loss_marginalized

def train_vaeac(X_train_miss, imputation_method='mean', epochs=40, batch_size=64):
    """Entraîne le modèle VAEAC complet avec une méthode d'imputation spécifique."""
    input_dim = X_train_miss.shape[1]

    # --- A. Imputation Initiale (Pré-traitement pour Full Encoder) ---
    if imputation_method == 'mean':
        imp = SimpleImputer(strategy='mean')
        X_filled = imp.fit_transform(X_train_miss)
    elif imputation_method == 'iterative':
        # MissForest approximation
        estimator = RandomForestRegressor(n_jobs=-1, max_depth=5)
        imp = IterativeImputer(estimator=estimator, max_iter=5, random_state=42)
        X_filled = imp.fit_transform(X_train_miss)
    else: # Zero
        X_filled = np.nan_to_num(X_train_miss, nan=0.0)

    # Conversion en tenseurs
    X_tensor = torch.FloatTensor(X_filled).to(DEVICE)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- B. Initialisation Modèle ---
    model = VAEAC_Network(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mask_gen = MaskGenerator(p=0.5)

    # --- C. Boucle d'entraînement ---
    model.train()
    print(f"--> Entraînement VAEAC (Init: {imputation_method.upper()})...")

    for epoch in tqdm(range(epochs)):
        for batch in loader:
            x_batch = batch[0]

            # Masque d'entraînement
            mask = mask_gen(x_batch).to(DEVICE)

            # x_complete : Entrée Full Encoder (imputation initiale)
            x_complete = x_batch

            # x_masked : Entrée Masked Encoder (zéros aux endroits masqués)
            x_masked = x_batch.clone()
            x_masked[mask.bool()] = 0

            optimizer.zero_grad()

            # Forward & Loss
            outs = model(x_complete, x_masked, mask)
            loss = vaeac_loss(x_complete, *outs, mask)

            loss.backward()
            optimizer.step()

    return model

def train_vaeac_synthetic(X_in, epochs=30):
    """
    Entraîne un nouveau VAEAC spécifique aux dimensions de ce dataset synthétique.
    Utilise les classes VAEAC_Network et vaeac_loss déjà définies plus haut.
    """
    dim = X_in.shape[1]
    model = VAEAC_Network(input_dim=dim, width=32, depth=3, latent_dim=8).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # On remplit les trous par 0 pour l'entrée du réseau (ou moyenne, peu importe ici)
    X_filled = np.nan_to_num(X_in, nan=0.0)
    tensor_x = torch.FloatTensor(X_filled).to(DEVICE)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=64, shuffle=True)

    model.train()
    for ep in range(epochs):
        for batch in loader:
            x_batch = batch[0]

            # Génération d'un masque d'entraînement (Bernoulli 0.5)
            # On réutilise votre MaskGenerator s'il est défini, sinon on fait simple:
            mask = torch.bernoulli(torch.full_like(x_batch, 0.5)).to(DEVICE)

            # Entrée masquée
            x_masked = x_batch.clone()
            x_masked[mask.bool()] = 0

            optimizer.zero_grad()
            # Forward pass (utilise votre classe VAEAC_Network)
            outs = model(x_batch, x_masked, mask)

            # Calcul de la loss (utilise votre fonction vaeac_loss)
            # outs contient : rec_mu, rec_logvar, f_mu, f_logvar, m_mu, m_logvar
            loss = vaeac_loss(x_batch, *outs, mask)

            loss.backward()
            optimizer.step()
    return model

def train_comparative_models(X_miss, epochs=30, batch_size=64):
    input_dim = X_miss.shape[1]

    # --- 1. PRÉPARATION DES DONNÉES ---

    # A. Approche MissForest
    print("🌲 [Modèle A] Préparation Imputation MissForest...")
    # On utilise RandomForest pour imputer les données manquantes avant l'entraînement
    rf_imputer = IterativeImputer(estimator=RandomForestRegressor(max_depth=5, n_jobs=-1), max_iter=5)
    X_imp_rf = rf_imputer.fit_transform(X_miss)

    # B. Approche Marginalisation (Théorique)
    # On remplace les NaNs par 0, mais on crée le Masque I pour la Loss
    X_zero = np.nan_to_num(X_miss, nan=0.0)
    Mask_I = (~np.isnan(X_miss)).astype(float) # 1 si observé, 0 si manquant

    # --- 2. INITIALISATION ---
    # CORRECTION ICI : On utilise le bon nom de classe 'VAEAC_Network'
    model_rf = VAEAC_Network(input_dim=input_dim).to(DEVICE)
    model_marg = VAEAC_Network(input_dim=input_dim).to(DEVICE)

    opt_rf = torch.optim.Adam(model_rf.parameters(), lr=1e-3)
    opt_marg = torch.optim.Adam(model_marg.parameters(), lr=1e-3)

    # On instancie le générateur de masque ici
    mask_gen = MaskGenerator(p=0.5)

    dataset_rf = TensorDataset(torch.FloatTensor(X_imp_rf))
    dataset_marg = TensorDataset(torch.FloatTensor(X_zero), torch.FloatTensor(Mask_I))

    loader_rf = DataLoader(dataset_rf, batch_size=batch_size, shuffle=True)
    loader_marg = DataLoader(dataset_marg, batch_size=batch_size, shuffle=True)

    history = {'loss_rf': [], 'loss_marg': []}

    # --- 3. BOUCLE ---
    print(f"🚀 Démarrage du duel sur {epochs} époques...")

    for ep in tqdm(range(epochs)):
        # -- Train MissForest --
        total_loss = 0
        for (batch_x,) in loader_rf:
            batch_x = batch_x.to(DEVICE)
            opt_rf.zero_grad()

            # Masque de tâche (Bernoulli 0.5)
            mask_task = mask_gen(batch_x).to(DEVICE)

            # Entrées du réseau
            x_complete = batch_x
            x_masked = batch_x.clone()
            x_masked[mask_task.bool()] = 0

            # Forward
            outs = model_rf(x_complete, x_masked, mask_task)
            # Loss Standard (utilise vaeac_loss définie plus haut)
            loss = vaeac_loss(x_complete, *outs, mask_task)

            loss.backward()
            opt_rf.step()
            total_loss += loss.item()
        history['loss_rf'].append(total_loss / len(loader_rf))

        # -- Train Marginalisé --
        total_loss = 0
        for (batch_x, batch_i) in loader_marg:
            batch_x, batch_i = batch_x.to(DEVICE), batch_i.to(DEVICE)
            opt_marg.zero_grad()

            mask_task = mask_gen(batch_x).to(DEVICE)

            x_complete = batch_x
            x_masked = batch_x.clone()
            x_masked[mask_task.bool()] = 0

            # Forward
            # Note: VAEAC_Network retourne (rec_mu, rec_logvar, f_mu, f_logvar, m_mu, m_logvar)
            # On récupère 'rec_mu' (la reconstruction) et 'rec_logvar' pour la loss
            rec_mu, rec_logvar, f_mu, f_logvar, m_mu, m_logvar = model_marg(x_complete, x_masked, mask_task)

            # On recrée la distribution pour la passer à la loss marginalisée
            rec_std = torch.exp(0.5 * rec_logvar)
            recon_dist = torch.distributions.Normal(rec_mu, rec_std)

            # Loss Marginalisée (ignore les trous via batch_i)
            # Attention : on passe recon_dist et les autres params
            loss = vaeac_loss_marginalized(
                x_target=batch_x,
                recon_dist=recon_dist,
                z=None, # Pas utilisé dans la loss actuelle
                mu_q=m_mu, logvar_q=m_logvar,
                mu_p=f_mu, logvar_p=f_logvar,
                mask_task=mask_task,
                mask_i=batch_i
            )

            loss.backward()
            opt_marg.step()
            total_loss += loss.item()
        history['loss_marg'].append(total_loss / len(loader_marg))

    return model_rf, model_marg, history
