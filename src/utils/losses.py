import torch
import numpy as np

def vaeac_loss(x_target, rec_mu, rec_logvar, f_mu, f_logvar, m_mu, m_logvar, mask):
    """
    Calcule la perte VAEAC complète avec régularisation.
    """
    # 1. Reconstruction Loss (NLL Gaussienne)
    rec_std = torch.exp(0.5 * rec_logvar)
    dist = torch.distributions.Normal(rec_mu, rec_std)
    log_prob = dist.log_prob(x_target)

    # On maximise la proba sur tout le vecteur (ou pondéré par le masque selon les variantes)
    recon_loss = -torch.mean(torch.sum(log_prob, dim=1))

    # 2. Divergence KL (Full Encoder || Masked Encoder)
    f_std = torch.exp(0.5 * f_logvar)
    m_std = torch.exp(0.5 * m_logvar)

    p_dist = torch.distributions.Normal(f_mu, f_std)
    q_dist = torch.distributions.Normal(m_mu, m_std)

    kl_loss = torch.mean(torch.sum(torch.distributions.kl_divergence(p_dist, q_dist), dim=1))

    # 3. Régularisation (Prior in Latent Space - Appendix C.3.1)
    # Empêche mu_psi et sigma_psi de diverger vers l'infini
    sigma_mu = 1e4
    sigma_sigma = 1e4

    reg_mu = torch.mean(torch.sum(m_mu**2, dim=1)) / (2 * sigma_mu**2)
    # Approximation du terme Gamma prior
    reg_sigma = torch.mean(torch.sum(torch.exp(m_logvar) - m_logvar, dim=1)) / sigma_sigma

    total_loss = recon_loss + kl_loss + reg_mu + reg_sigma
    return total_loss

def vaeac_loss_marginalized(x_target, recon_dist, z, mu_q, logvar_q, mu_p, logvar_p, mask_task, mask_i):
    """
    Calcule l'ELBO en ignorant les erreurs de reconstruction sur les données
    qui étaient manquantes à l'origine (indiquées par mask_i = 0).
    """

    # 1. Terme de Reconstruction (Log-Likelihood)
    log_prob = recon_dist.log_prob(x_target)

    # --- C'EST ICI QUE TOUT CHANGE ---
    # On applique le masque I pour ne garder que les erreurs sur les vraies données
    # On ne punit pas le modèle s'il se trompe sur une valeur qui n'existe pas.
    log_prob = log_prob * mask_i

    # On somme uniquement sur les dimensions valides
    # Normalisation par le nombre de valeurs observées pour garder l'échelle stable
    recon_loss = -torch.sum(log_prob) / (torch.sum(mask_i) + 1e-8)

    # 2. Terme de Régularisation (KL Divergence)
    std_q = torch.exp(0.5 * logvar_q)
    std_p = torch.exp(0.5 * logvar_p)
    kld = torch.log(std_p + 1e-8) - torch.log(std_q + 1e-8) + (std_q**2 + (mu_q - mu_p)**2) / (2 * std_p**2 + 1e-8) - 0.5

    # La KLD est calculée globalement (sur l'espace latent), le masque I n'intervient pas ici
    kld_loss = torch.mean(torch.sum(kld, dim=1))

    return recon_loss + kld_loss
