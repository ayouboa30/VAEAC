import torch

class MaskGenerator:
    """Génère des masques aléatoires (Bernoulli) pour l'entraînement."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, batch):
        # 1 = Manquant, 0 = Observé
        return torch.bernoulli(torch.full_like(batch, self.p))
