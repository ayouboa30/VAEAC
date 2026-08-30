import numpy as np
from scipy.stats import entropy

def calculate_kl_divergence(x_true, x_pred, n_bins=50):
    kl_scores = []
    eps = 1e-8
    for i in range(x_true.shape[1]):
        min_val = min(x_true[:, i].min(), x_pred[:, i].min())
        max_val = max(x_true[:, i].max(), x_pred[:, i].max())
        bins = np.linspace(min_val, max_val, n_bins)
        p_true, _ = np.histogram(x_true[:, i], bins=bins, density=True)
        p_pred, _ = np.histogram(x_pred[:, i], bins=bins, density=True)
        p_true += eps
        p_pred += eps
        p_true /= p_true.sum()
        p_pred /= p_pred.sum()
        kl = entropy(p_true, p_pred)
        kl_scores.append(kl)
    return np.mean(kl_scores)
