### the analytical
import torch
import torch.nn as nn
import numpy as np
import math


class FastDCT2D(nn.Module):
    """
    Production-ready DCT-II layer using cached matrix multiplication.
    Optimized for GPU throughput on typical Deep Learning image sizes (< 128x128).
    """
    def __init__(self, height, width, device='cuda'):
        super().__init__()
        self.height = height
        self.width = width
        self.device = device
        
        # Precompute and cache the orthogonal DCT matrices
        # We register them as buffers so they are saved with the model state_dict
        # but are not treated as trainable parameters.
        self.register_buffer('dct_h', self._make_dct_matrix(height))
        self.register_buffer('dct_w', self._make_dct_matrix(width))

    def _make_dct_matrix(self, N):
        """Constructs the orthonormal DCT-II basis matrix."""
        # Grid of n (time/pixel indices) and k (frequency indices)
        n = torch.arange(N, device=self.device).float()
        k = torch.arange(N, device=self.device).float()
        
        # DCT-II argument: pi/N * (n + 0.5) * k
        # Shape: [N, N] where rows are k, cols are n
        args = (math.pi / N) * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1)
        matrix = torch.cos(args)
        
        # Orthonormalization scale factors
        # k=0: sqrt(1/N), k>0: sqrt(2/N)
        scale = torch.ones(N, device=self.device)
        scale[0] = 1.0 / math.sqrt(2.0)
        scale *= math.sqrt(2.0 / N)
        
        # Broadcast scale across rows (k)
        return scale.unsqueeze(1) * matrix

    def forward(self, x):
        """
        Forward DCT: Pixel -> Frequency
        Formula: Y = C_h * X * C_w^T
        """
        # x: [Batch, Channels, Height, Width]
        # Apply DCT to Height dimension (dim 2)
        # We use matrix multiplication: M @ x
        # Since x is [B, C, H, W], we treat H as the column vector for the transform
        # matmul broadcasts over B, C. 
        # C_h is [H, H], x is [..., H, W]. Result is [..., H, W]
        x_freq_h = torch.matmul(self.dct_h, x)
        
        # Apply DCT to Width dimension (dim 3)
        # We need to act on the last dimension. 
        # Standard matmul A @ B does matrix mul on last two dims.
        # We want X @ C_w.T
        out = torch.matmul(x_freq_h, self.dct_w.T)
        return out

    def inverse(self, x):
        """
        Inverse DCT: Frequency -> Pixel
        Formula: X = C_h^T * Y * C_w (since C is orthogonal, C^-1 = C^T)
        """
        # Undo Height transform: C_h^T @ x
        x_pixel_h = torch.matmul(self.dct_h.T, x)
        
        # Undo Width transform: X @ C_w
        out = torch.matmul(x_pixel_h, self.dct_w)
        return out



class DCT2D(nn.Module):
    def __init__(self, height, width, device='cuda'):
        super().__init__()
        self.height = height
        self.width = width
        self.device = device
        
        # Precompute Phase Factors for Forward/Inverse Correction
        # These correct the grid alignment from "Symmetric Extension" to "DCT Grid"
        self.register_buffer('phase_h', self._get_phase(height).unsqueeze(1)) # [H, 1]
        self.register_buffer('phase_w', self._get_phase(width).unsqueeze(0))  # [1, W]
        
        # Scaling factors (Orthonormality)
        self.register_buffer('scale_h', self._get_scale(height).unsqueeze(1))
        self.register_buffer('scale_w', self._get_scale(width).unsqueeze(0))

    def _get_phase(self, N):
        # exp(-i * pi * k / (2N))
        k = torch.arange(N, device=self.device).float()
        return torch.exp(-1j * math.pi * k / (2 * N))

    def _get_scale(self, N):
        # Standard Ortho Scale: sqrt(1/N) for DC, sqrt(2/N) for AC
        s = torch.ones(N, device=self.device)
        s[0] = 1.0 / math.sqrt(2.0)
        s *= math.sqrt(2.0 / N)
        return s

    def forward(self, x):
        """
        Forward DCT via FFT:
        1. Symmetrically extend signal (2N)
        2. Compute FFT
        3. Apply Phase Shift (to correct for half-sample symmetry)
        """
        # 1. Symmetric Extension: [x, flip(x)]
        # Height Dim
        x = torch.cat([x, x.flip([2])], dim=2)
        # Width Dim
        x = torch.cat([x, x.flip([3])], dim=3)
        
        # 2. FFT2 (Real -> Complex)
        x_fft = torch.fft.fft2(x)
        
        # 3. Crop to N x N (The rest is redundant due to symmetry)
        x_fft = x_fft[:, :, :self.height, :self.width]
        
        # 4. Phase Correction & Scaling
        # We need to broadcast the phase factors: H uses phase_h, W uses phase_w
        # Phase[u,v] = Phase_H[u] * Phase_W[v]
        phase = self.phase_h * self.phase_w
        
        # The result of (FFT * Phase) is Real. We take .real to remove numerical noise.
        return (x_fft * phase).real * self.scale_h * self.scale_w


    def inverse(self, x):
        """
        Inverse DCT via FFT (IDCT):
        1. Undo Scaling & Phase Shift
        2. Reconstruct Hermitian Symmetric Spectrum
        3. Inverse FFT
        """
        # 1. Undo Scaling
        x = x / (self.scale_h * self.scale_w)
        
        # 2. Undo Phase Shift (Multiply by Conjugate Phase)
        # We treat inputs as Complex (Real part = x, Imag = 0)
        phase_inv = torch.conj(self.phase_h * self.phase_w)
        z = torch.complex(x, torch.zeros_like(x)) * phase_inv
        
        # 3. Reconstruct 2N Symmetric Spectrum for IFFT
        # We need to pad z to [2H, 2W] such that it has Hermitian symmetry
        # (This ensures the IFFT output is purely Real)
        
        # This step is computationally tricky in PyTorch 
        # (padding with complex conjugates mirrored).
        # A simplified approach for IDCT using fft.irfft is:
        # Reconstruct the "Real-Symmetric" signal in frequency domain.
        
        # Efficient Shortcut: The 'irfft' function expects a specific half-spectrum.
        # We can map DCT coeffs directly to the format 'irfft' expects.
        # However, implementing the exact padding for 2D irfft is verbose.
        
        # --- Fallback to Separable 1D FFTs for Clarity & Correctness ---
        # Inverse Height
        x = self._idct_1d(x, dim=2, phase=self.phase_h, N=self.height)
        # Inverse Width
        x = self._idct_1d(x, dim=3, phase=self.phase_w, N=self.width)
        return x

    def _idct_1d(self, x, dim, phase, N):
        """
        Helper for 1D IDCT via FFT.
        Formula: x = IRFFT( V ) where V is constructed from X.
        """
        # 1. Undo Phase
        # Treat x as real part of complex signal
        x_complex = torch.complex(x, torch.zeros_like(x)) * torch.conj(phase)
        
        # 2. Construct Input for IRFFT (Length N+1 for output 2N)
        # irfft takes (N+1) complex coeffs and produces 2N real samples
        # We have N coeffs. We need to pad the Nyquist freq (usually 0).
        
        # Pad 1 zero at the end of the dimension
        pad_shape = [0, 0] * x.ndim
        pad_idx = (x.ndim - 1 - dim) * 2 + 1 # Pad 'right' side of dim
        pad_shape[pad_idx] = 1
        
        # Apply padding [B, ..., N, ...] -> [B, ..., N+1, ...]
        z_padded = torch.nn.functional.pad(x_complex, pad_shape)
        
        # 3. Compute IRFFT (Result is 2N length)
        # n=2*N ensures we get length 2N output
        out = torch.fft.irfft(z_padded, n=2*N, dim=dim)
        
        # 4. Crop to N (The 2N extension was symmetric padding)
        # Slice object to crop the specific dimension
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(0, N)
        
        return out[tuple(slices)]




class DecoupledOTCFM(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), device='cuda', r_low=0.2, 
                 operators=['freq', 'color', 'spatial'], 
                 decay_mode='piecewise_linear', # Choose 'exponential', 'truncated', 'piecewise_linear
                 k_base=6.0, k_beta=0.7, # Parameters for exponential mode
                 t0_base=0.4, t0_beta=0.7, s_steep=30.0,
                 t1_base=0.8): # Parameters for truncated mode
        
        super().__init__()
        self.device = device
        self.C, self.H, self.W = img_shape
        self.operators = operators
        self.decay_mode = decay_mode
        self.s_steep = s_steep
        
        ## Compute the Geometric Separation of Timescales Internally 
        self.op_params = {}
        
        if self.decay_mode in ['exponential', 'piecewise_linear']:
            current_param = k_base
            for op in self.operators:
                self.op_params[op] = current_param
                current_param = current_param * k_beta  # Geometric scaling of decay rate
        
        elif self.decay_mode == 'truncated':
            current_param = t0_base
            for op in self.operators:
                self.op_params[op] = current_param
                current_param = current_param * t0_beta # Geometric scaling of the drop-out horizon
        
        elif self.decay_mode == 'hard_truncated_exponential':
            # This mode tracks BOTH the exponential decay 'k' and the hard cutoff 't1'
            curr_k = k_base
            for op in self.operators:
                self.op_params[op] = {'k': curr_k, 't1': t1_base}
                curr_k = curr_k * k_beta
        
        else:
            raise ValueError(f"Unknown decay_mode: {decay_mode}")
        
        ## Map the config strings to the actual python methods
        self.op_map = {
            'freq': self.op_freq,
            'color': self.op_color, 
            'spatial': self.op_spatial,
            'illumination': self.op_illumination 
        }
        
        # --- Operator Bases Initialization ---
        self.dct = FastDCT2D(self.H, self.W, device)
        
        # 1. Orthogonal Color Transformation (W)
        self.rgb_to_yuv = torch.tensor([
            [1/math.sqrt(3),  1/math.sqrt(3),  1/math.sqrt(3)], # Content: Luma (Y)
            [1/math.sqrt(2), -1/math.sqrt(2),  0.0           ], # Style: Chroma 1
            [1/math.sqrt(6),  1/math.sqrt(6), -2/math.sqrt(6)]  # Style: Chroma 2
        ], device=device)
        self.yuv_to_rgb = self.rgb_to_yuv.T
        
        # 2. Orthogonal Frequency Projections
        y, x = torch.meshgrid(torch.arange(self.H, device=device), 
                            torch.arange(self.W, device=device), indexing='ij')
        self.freq_dist = torch.sqrt((x / self.W)**2 + (y / self.H)**2) / math.sqrt(2)
        r_low_normalized = r_low / self.W
        
        gaussian_mask = torch.exp(- (self.freq_dist ** 2) / (2 * (r_low_normalized ** 2)))
        self.freq_P_c = gaussian_mask.unsqueeze(0).unsqueeze(0).float()
        self.freq_P_s = 1.0 - self.freq_P_c
        
        # 3. Orthogonal Spatial Projections
        self.patch_size = 8
        h_patches, w_patches = self.H // self.patch_size, self.W // self.patch_size
        torch.manual_seed(42)
        mask_grid = (torch.rand((1, 1, h_patches, w_patches), device=device) > 0.4).float()
        self.spatial_mask = torch.nn.functional.interpolate(mask_grid, size=(self.H, self.W), mode='nearest')
        self.spatial_P_c = 1.0 - self.spatial_mask 
        self.spatial_P_s = self.spatial_mask


    def _get_curriculum_rates(self, t, param):
        """
        Computes the NORMALIZED survival rates: (P_c + \alpha(t) P_s)
        and their exact analytical time derivatives.
        """
        gamma_norm = torch.ones_like(t)
        dot_gamma_norm = torch.zeros_like(t) 
        
        if self.decay_mode == 'exponential':
            k = param
            alpha_norm = torch.exp(-k * t)
            dot_alpha_norm = -k * alpha_norm
        
        elif self.decay_mode == 'piecewise_linear':
            k = param
            # f(t) = max(0, 1 - kt)
            alpha_norm = torch.clamp(1.0 - k * t, min=0.0)
            
            # dot_f(t) = -k if t < 1/k else 0
            dot_alpha_norm = torch.where(t < (1.0 / k), -k * torch.ones_like(t), torch.zeros_like(t))
        
        elif self.decay_mode == 'truncated':
            t0 = param
            # Precompute constant C to ensure alpha(0) = 1.0
            C = 1.0 + math.exp(-self.s_steep * t0)
            
            # The exponential term inside the sigmoid
            exp_inner = torch.exp(self.s_steep * (t - t0))
            denom = 1.0 + exp_inner
            
            # Forward Schedule
            alpha_norm = C / denom
            
            # Exact Analytical Derivative (Quotient Rule)
            dot_alpha_norm = (-self.s_steep * C * exp_inner) / (denom ** 2)
        
        elif self.decay_mode == 'hard_truncated_exponential':
            # Extract both parameters from the dictionary
            k = param['k']
            t1 = param['t1']
            
            # 1. Evaluate the standard exponential state and derivative
            exp_decay = torch.exp(-k * t)
            dot_exp_decay = -k * exp_decay
            
            # 2. Apply strict piecewise truncation using torch.where
            # If t < t1, use the exponential curve. Otherwise, use strict 0.0.
            alpha_norm = torch.where(t < t1, exp_decay, torch.zeros_like(t))
            
            # 3. Apply the same piecewise logic to the velocity.
            # (Note: Mathematically drops the infinite Dirac delta at t=t1)
            dot_alpha_norm = torch.where(t < t1, dot_exp_decay, torch.zeros_like(t))
            
        return gamma_norm, alpha_norm, dot_gamma_norm, dot_alpha_norm


    # --- Operator Methods ---
    def op_freq(self, x, dx, t, param):
        gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, param)
        scale_x  = gamma * self.freq_P_c + alpha * self.freq_P_s
        scale_dx = dot_gamma * self.freq_P_c + dot_alpha * self.freq_P_s
        
        X_freq = self.dct(x)
        dX_freq = self.dct(dx)
        
        new_X_freq = X_freq * scale_x
        new_dX_freq = X_freq * scale_dx + dX_freq * scale_x
        
        return self.dct.inverse(new_X_freq), self.dct.inverse(new_dX_freq)

    def op_color(self, x, dx, t, param):
        gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, param)
        x_perm = x.permute(0, 2, 3, 1)
        dx_perm = dx.permute(0, 2, 3, 1)
        
        yuv_x = torch.matmul(x_perm, self.rgb_to_yuv.T)
        yuv_dx = torch.matmul(dx_perm, self.rgb_to_yuv.T)
        
        B = x.shape[0]
        scale_x = torch.cat([gamma, alpha, alpha], dim=-1).view(B, 1, 1, 3)
        scale_dx = torch.cat([dot_gamma, dot_alpha, dot_alpha], dim=-1).view(B, 1, 1, 3)
        
        new_yuv_x = yuv_x * scale_x
        new_yuv_dx = yuv_x * scale_dx + yuv_dx * scale_x
        
        new_x = torch.matmul(new_yuv_x, self.yuv_to_rgb.T).permute(0, 3, 1, 2)
        new_dx = torch.matmul(new_yuv_dx, self.yuv_to_rgb.T).permute(0, 3, 1, 2)
        return new_x, new_dx

    def op_spatial(self, x, dx, t, param):
        gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, param)
        scale_x = gamma * self.spatial_P_c + alpha * self.spatial_P_s
        scale_dx = dot_gamma * self.spatial_P_c + dot_alpha * self.spatial_P_s
        
        new_x = x * scale_x
        new_dx = x * scale_dx + dx * scale_x
        return new_x, new_dx

    def op_illumination(self, x, dx, t, param):
        gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, param)
        mean_x = torch.mean(x, dim=(2, 3), keepdim=True)
        mean_dx = torch.mean(dx, dim=(2, 3), keepdim=True)
        
        contrast_x = x - mean_x
        contrast_dx = dx - mean_dx
        
        new_x = gamma * contrast_x + alpha * mean_x
        new_dx = dot_gamma * contrast_x + gamma * contrast_dx + dot_alpha * mean_x + alpha * mean_dx
        return new_x, new_dx


    # --- The Core CFM Computation ---
    def compute_target(self, x0, t):
        ##* t has shape [B, 1, 1, 1]
        
        s_curr = x0
        ds_curr = torch.zeros_like(x0) 
        
        for op_name in self.operators:
            op_func = self.op_map[op_name]
            param_val = self.op_params[op_name]
            s_curr, ds_curr = op_func(s_curr, ds_curr, t, param_val)
            
        C_t_x0 = s_curr
        dot_C_t_x0 = ds_curr
        
        M_t_x0 = (1 - t) * C_t_x0
        dot_M_t_x0 = -1.0 * C_t_x0 + (1 - t) * dot_C_t_x0
        
        x_1 = torch.randn_like(x0)
        x_t = M_t_x0 + t * x_1
        u_t = dot_M_t_x0 + x_1
        
        return x_t, u_t



#####* Optimal Flow Control Case
class OTFlow(nn.Module):
    """
    Standard Optimal Transport Conditional Flow Matching.
    Path: Straight line from Data (x0) to Noise (x1).
    Formula: x_t = (1 - t) * x0 + t * x1
    Target:  u_t = x1 - x0
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def compute_target(self, x0, t):
        """
        Computes x_t and u_t for the OT path.
        
        Args:
            x0: Clean image batch [B, C, H, W]
            t: Time batch [B, 1, 1, 1]
        """
        # 1. Sample Noise x1 ~ N(0, I)
        x1 = torch.randn_like(x0)
        
        # 2. Compute State x_t (Linear Interpolation)
        # x_t = (1 - t) * x0 + t * x1
        x_t = (1 - t) * x0 + t * x1
        
        # 3. Compute Target Velocity u_t
        # u_t = d/dt (x_t) = x1 - x0
        u_t = x1 - x0
        
        return x_t, u_t