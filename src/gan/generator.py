import torch
import torch.nn as nn

class GeneratorMLP(nn.Module):
    """Generator for conditional GAN.
    Input: Noise z (dim 64) and Condition (label).
    Output: Synthetic feature vector.
    """
    def __init__(self, feature_dim: int, noise_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + 1, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim),
            nn.Tanh() # Features should be normalized to [-1, 1] before GAN
        )

    def forward(self, z, label):
        """
        z: FloatTensor of shape (batch, noise_dim)
        label: FloatTensor of shape (batch, 1)
        """
        x = torch.cat([z, label], dim=1)
        return self.model(x)
