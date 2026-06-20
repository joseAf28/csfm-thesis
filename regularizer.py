import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentRegularizer(nn.Module):
    """
    A unified class for latent space regularization supporting:
    1. SIGReg (LeJEPA): Enforces Isotropic Gaussian distribution via random projections.
    2. VICReg: Enforces unit variance and decorrelation (Covariance) explicitly.
    3. L1: Simple sparsity regularization (L1 norm).
    
    Args:
        feature_dim (int): Dimension of the input embeddings z.
        mode (str): 'sigreg', 'vicreg', or 'l1'.
        reduce_mean (bool): If False, scales the loss by feature_dim to match summed task losses.
        # SIGReg Hyperparameters
        num_projections (int): Number of random slices (M). Default: 1024.
        range_t (float): Integration range [-T, T] for Epps-Pulley. Default: 5.0.
        num_t (int): Number of quadrature points. Default: 17.
        # VICReg Hyperparameters
        vicreg_var_weight (float): Weight for variance term. Default: 25.0.
        vicreg_cov_weight (float): Weight for covariance term. Default: 1.0.
    """
    def __init__(self, feature_dim, mode='sigreg', reduce_mean=True,
                feature_dim_ref=128, # widen_factor = 2 -> 128
                num_projections=256, range_t=3.0, num_t=17,
                vicreg_var_weight=25.0, vicreg_cov_weight=1.0):
        super().__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        self.reduce_mean = reduce_mean  # Tracks the reduction strategy of the main loss
        
        self.dim_scalar = feature_dim_ref / feature_dim if not reduce_mean else 1.0
        
        # --- SIGReg Init ---
        if self.mode == 'sigreg':
            self.num_projections = num_projections
            self.register_buffer('t_eval', torch.linspace(-range_t, range_t, num_t))
            self.register_buffer('target_cf', torch.exp(-0.5 * self.t_eval**2))
            
            self.num_projections = num_projections
            
            # 1. Define integration grid from 0 to 3.0
            t = torch.linspace(0, range_t, num_t, dtype=torch.float32)
            dt = range_t / (num_t - 1)
            
            # 2. Precompute the 2x Trapezoidal weights for the symmetric (-inf, inf) integral
            weights = torch.full((num_t,), 2 * dt, dtype=torch.float32)
            weights[[0, -1]] = dt
            
            # 3. Precompute Gaussian window
            window = torch.exp(-0.5 * t.square())
            
            # 4. Register buffers (t, phi, and combined weights)
            self.register_buffer('t_eval', t)
            self.register_buffer('window', window)
            self.register_buffer('weights', weights * window)
            
        # --- VICReg Init ---
        elif self.mode == 'vicreg':
            self.var_weight = vicreg_var_weight
            self.cov_weight = vicreg_cov_weight

    def forward(self, z):
        z = z - z.mean(dim=0)

        if self.mode == 'sigreg':
            loss = self._sigreg_loss(z)
        elif self.mode == 'vicreg':
            loss = self._vicreg_loss(z)
        elif self.mode == 'l1':
            loss = self._l1_loss(z)
        elif self.mode == 'zero':
            loss =  self._zero_loss(z)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return loss * self.dim_scalar


    def _sigreg_loss(self, z):
        B, D = z.shape
        
        # 1. Sample M random projection directions (normalized to unit hypersphere)
        # Using the official dimensions: [D, num_projections]
        A = torch.randn(D, self.num_projections, device=z.device)
        A = A.div_(A.norm(p=2, dim=0)) 
        
        # 2. Project embeddings and multiply by t
        # z: [B, D] @ A: [D, M] -> [B, M]
        # unsqueeze -> [B, M, 1] * t_eval: [num_t] -> x_t: [B, M, num_t]
        x_t = (z @ A).unsqueeze(-1) * self.t_eval
        
        # 3. Compute the ECF Error squared: | ECF(t) - phi_G(t) |^2
        # Mean over the batch dimension (dim=0)
        ecf_real = x_t.cos().mean(dim=0)  # [M, num_t]
        ecf_imag = x_t.sin().mean(dim=0)  # [M, num_t]
        
        err = (ecf_real - self.window).square() + ecf_imag.square() # [M, num_t]
        
        # 4. Integrate using precomputed weights AND scale by Batch Size (B)
        # err @ weights performs the integration over num_t
        statistic = (err @ self.weights) * B  # [M]
        
        # 5. Expectation over the M random projections
        base_loss = statistic.mean()
        
        # Scale for sum reduction if required by your pipeline
        if not self.reduce_mean:
            return base_loss * (0.5 * self.feature_dim)
        return base_loss


    def _vicreg_loss(self, z):
        B, D = z.shape
        
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        
        # Sum over latent dims if reduce_mean is False
        if not self.reduce_mean:
            std_loss = torch.sum(F.relu(1 - std_z))
        else:
            std_loss = torch.mean(F.relu(1 - std_z))
            
        cov_z = (z.T @ z) / (B - 1)
        off_diag_cov = cov_z.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
        
        # Note: VICReg covariance is usually scaled by 1/D. We drop the /D for sum reduction.
        if not self.reduce_mean:
            cov_loss = off_diag_cov.pow(2).sum()
        else:
            cov_loss = off_diag_cov.pow(2).sum() / D
        
        return self.var_weight * std_loss + self.cov_weight * cov_loss


    def _l1_loss(self, z):
        # Mean over batch (dim=0), Sum over features (dim=1) if reduce_mean=False
        if not self.reduce_mean:
            return torch.abs(z).mean(dim=0).sum() * 0.5
        return torch.abs(z).mean()
    
    def _zero_loss(self, z):
        return torch.tensor(0.0, device=z.device)