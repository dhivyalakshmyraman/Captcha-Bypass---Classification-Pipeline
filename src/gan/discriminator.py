import torch
import torch.nn as nn

class DiscriminatorMLP(nn.Module):
    """Discriminator for conditional GAN.
    Input: Feature vector and Condition (label).
    Output: Probability of being real (1) or fake (0).
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(feature_dim + 1, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        """
        x: FloatTensor of shape (batch, feature_dim)
        label: FloatTensor of shape (batch, 1)
        """
        combined = torch.cat([x, label], dim=1)
        return self.model(combined)
