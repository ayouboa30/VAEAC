import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipBlock(nn.Module):
    """Bloc linéaire simple avec activation LeakyReLU."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2)  # Activation LeakyReLU comme spécifié
        )

    def forward(self, x):
        return self.net(x)

class VAEAC_Network(nn.Module):
    def __init__(self, input_dim, width=32, depth=3, latent_dim=8):
        super().__init__()
        self.input_dim = input_dim

        # --- 1. Masked Encoder (Prior Network) p_psi ---
        # Input: [Features (masked=0)] + [Mask] -> taille 2 * input_dim
        self.me_input = nn.Linear(input_dim * 2, width)
        self.me_blocks = nn.ModuleList([SkipBlock(width, width) for _ in range(depth)])
        self.me_mu = nn.Linear(width, latent_dim)
        self.me_logvar = nn.Linear(width, latent_dim)

        # --- 2. Full Encoder (Proposal Network) p_phi ---
        # Input: [Features (complets)] + [Mask]
        self.fe_input = nn.Linear(input_dim * 2, width)
        self.fe_blocks = nn.ModuleList([SkipBlock(width, width) for _ in range(depth)])
        self.fe_mu = nn.Linear(width, latent_dim)
        self.fe_logvar = nn.Linear(width, latent_dim)

        # --- 3. Decoder (Generative Network) p_theta ---
        # Input: [z]
        self.dec_input = nn.Linear(latent_dim, width)
        # Input des blocs: [Sortie précédente] + [Skip connection du Masked Encoder] (Concaténation)
        self.dec_blocks = nn.ModuleList([SkipBlock(width * 2, width) for _ in range(depth)])

        self.dec_mu = nn.Linear(width, input_dim)
        self.dec_logvar = nn.Linear(width, input_dim)

    def reparameterize(self, mu, logvar):
        """Trick de reparamétrisation : z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_masked_encoder(self, x_masked, mask):
        # Concaténation données + masque
        x = torch.cat([x_masked, mask], dim=1)
        x = F.leaky_relu(self.me_input(x), 0.2)

        skips = []
        for block in self.me_blocks:
            x = block(x)
            skips.append(x) # On sauvegarde l'activation pour le décodeur

        return self.me_mu(x), self.me_logvar(x), skips

    def forward_full_encoder(self, x_complete, mask):
        x = torch.cat([x_complete, mask], dim=1)
        x = F.leaky_relu(self.fe_input(x), 0.2)
        for block in self.fe_blocks:
            x = block(x)
        return self.fe_mu(x), self.fe_logvar(x)

    def forward_decoder(self, z, skips):
        x = F.leaky_relu(self.dec_input(z), 0.2)

        # Application des Skip-Connections par concaténation
        for i, block in enumerate(self.dec_blocks):
            # Concaténation : [Decoder State, Masked Encoder State]
            x = torch.cat([x, skips[i]], dim=1)
            x = block(x)

        return self.dec_mu(x), self.dec_logvar(x)

    def forward(self, x_complete, x_masked, mask):
        # 1. Masked Encoder (génère aussi les skips)
        m_mu, m_logvar, skips = self.forward_masked_encoder(x_masked, mask)
        # 2. Full Encoder
        f_mu, f_logvar = self.forward_full_encoder(x_complete, mask)
        # 3. Sampling (via Full Encoder pendant le training)
        z = self.reparameterize(f_mu, f_logvar)
        # 4. Decoder
        rec_mu, rec_logvar = self.forward_decoder(z, skips)

        return rec_mu, rec_logvar, f_mu, f_logvar, m_mu, m_logvar
