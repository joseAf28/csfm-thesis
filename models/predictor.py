import torch
import torch.nn as nn

class JEPAPredictor(nn.Module):
    """
    A lightweight MLP Predictor for LeJEPA.
    Maps the context representation z_C to the target representation z_T.
    """
    def __init__(self, feature_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, z_context):
        return self.net(z_context)