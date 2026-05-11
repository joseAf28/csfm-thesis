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


class TargetCompositeFlow(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), device='cuda', FastDCT=True, modes=None, operators=None):
        super().__init__()
        self.device = device
        self.C, self.H, self.W = img_shape
        if FastDCT:
            self.dct = FastDCT2D(self.H, self.W, device)
        else:
            self.dct = DCT2D(self.H, self.W, device)
        
        # 2. Modes Configuration
        if modes is None:
            self.modes = ['linear', 'linear', 'linear']
        else:
            self.modes = modes
        
        # 3. Operators Configuration
        # Mapping string names to functions
        dict_conv = {'freq': self.op_freq, 'color': self.op_color, 'mask': self.op_spatial}
        keys_dict_conv = list(dict_conv.keys())
        
        if operators is None:
            # Default list of function references
            self.operators = [self.op_freq, self.op_color, self.op_spatial]
        else: 
            self.operators = []
            for name in operators:
                assert name in keys_dict_conv, f"Mismatch operator: {name} is not in {keys_dict_conv}."
                self.operators.append(dict_conv[name])
        
        # Validation
        assert len(self.modes) == len(self.operators), f"Mismatch: {len(self.modes)} modes provided for {len(self.operators)} operators."
        
        # Orthonormal Color Transformation
        # Row 0: Luminance (Avg of RGB) - Normalized
        # Row 1: Red-Green Contrast
        # Row 2: Blue-Yellow Contrast
        self.rgb_to_yuv = torch.tensor([
            [1/math.sqrt(3),  1/math.sqrt(3),  1/math.sqrt(3)], # Luma (aligned with gray)
            [1/math.sqrt(2), -1/math.sqrt(2),  0.0           ], # Chroma 1 (R vs G)
            [1/math.sqrt(6),  1/math.sqrt(6), -2/math.sqrt(6)]  # Chroma 2 (RG vs B)
        ], device=device)
        

        self.yuv_to_rgb = self.rgb_to_yuv.T
        
        # Freq Grid
        y, x = torch.meshgrid(torch.arange(self.H, device=device), torch.arange(self.W, device=device), indexing='ij')
        self.freq_dist = torch.sqrt((x / self.W)**2 + (y / self.H)**2) / math.sqrt(2)
        
        # Spatial Patch Grid (Pre-computed for deterministic "high variance regions")
        # Creates 8x8 blocks
        self.patch_size = 8
        h_patches, w_patches = self.H // self.patch_size, self.W // self.patch_size
        # Create a fixed random mask pattern (1=Masked, 0=Visible)
        # In a real training loop, this might be sampled per batch, but fixed for t differentiation
        torch.manual_seed(42) # For reproducibility in visualization
        self.spatial_mask_grid = (torch.rand((1, 1, h_patches, w_patches), device=device) > 0.4).float()
        # Upsample to pixel size
        self.spatial_mask = torch.nn.functional.interpolate(self.spatial_mask_grid, size=(self.H, self.W), mode='nearest')


    def _get_warped_schedule(self, t, mode='linear'):
        """
        Applies time-warping gamma(t) then computes Cosine Schedule.
        Returns alpha, sigma, dot_alpha_total, dot_sigma_total
        """
        # 1. Apply Time Warping gamma(t)
        # t is [1, 1, 1, 1]
        if mode == 'fast_start':
            # Quadratic Ease-Out (1 - (1-t)^2)
            # Shape is very similar (concave down), but deriv at 0 is 2.0
            t_warped = 1.0 - torch.pow(1.0 - t, 2.0)
            dot_gamma = 2.0 * (1.0 - t)
            
        elif mode == 'delayed':
            # gamma(t) = t^2
            # dot_gamma = 2t
            t_warped = torch.pow(t, 2.0)
            dot_gamma = 2 * t
            
        else:
            t_warped = t
            dot_gamma = torch.ones_like(t)

        # Clamp for safety
        t_warped = torch.clamp(t_warped, 0.0, 1.0)

        # 2. Compute Schedule on Warped Time
        theta = (math.pi / 2) * t_warped
        alpha = torch.cos(theta)
        sigma = torch.sin(theta)
        
        # Derivatives w.r.t warped time
        d_alpha_d_twarped = -(math.pi / 2) * torch.sin(theta)
        d_sigma_d_twarped = (math.pi / 2) * torch.cos(theta)
        
        # 3. Chain Rule: d/dt = d/d_twarped * dot_gamma
        dot_alpha = d_alpha_d_twarped * dot_gamma
        dot_sigma = d_sigma_d_twarped * dot_gamma
        
        return alpha, sigma, dot_alpha, dot_sigma, t_warped


    # --- Operators with Specific Warping ---
    
    def op_freq(self, x, t, deriv=False, mode='linear', k_blur=10.0):
        """
        Accelerated Frequency Degradation with Time Derivative.
        Corrects the approximation by including the derivative of the blur mask.
        """
        # 1. Get Schedule Terms
        alpha, _, dot_alpha, _, t_warped = self._get_warped_schedule(t, mode=mode)
        
        # 2. Compute dot_gamma (Derivative of time warping)
        # We assume the same logic as _get_warped_schedule
        if mode == 'fast_start':
            # gamma(t) = 1 - (1-t)^2  -->  dot_gamma = 2(1-t)
            dot_gamma = 2.0 * (1.0 - t)
        elif mode == 'delayed':
            # gamma(t) = t^2  -->  dot_gamma = 2t
            dot_gamma = 2.0 * t
        else: 
            # gamma(t) = t  -->  dot_gamma = 1
            dot_gamma = torch.ones_like(t)

        # 3. Compute Blur Mask M(t)
        # bandwidth = k * gamma(t)
        bandwidth = k_blur * t_warped
        # M(t) = exp(-bandwidth * f^2)
        freq_sq = self.freq_dist ** 2
        mask_val = torch.exp(-bandwidth * freq_sq)
        
        if deriv:
            # PRODUCT RULE: d/dt (alpha * M) = dot_alpha * M + alpha * dot_M
            
            # Term A: Change in Envelope (dot_alpha * M)
            term_alpha = dot_alpha * mask_val
            
            # Term B: Change in Blur Width (alpha * dot_M)
            # dot_M = M * (-k * dot_gamma * f^2)
            dot_M = mask_val * (-k_blur * dot_gamma * freq_sq)
            term_blur = alpha * dot_M
            
            # Combine
            total_scale = term_alpha + term_blur
            
            return self.dct.inverse(self.dct(x) * total_scale)
        
        # Standard Forward: alpha * M
        return self.dct.inverse(self.dct(x) * mask_val * alpha)
    
    
    def op_color(self, x, t, deriv=False, mode="linear"):
        """Standard Linear Color Degradation (Batch Safe)"""
        # alpha, dot_alpha are [B, 1, 1, 1]
        alpha, _, dot_alpha, _, _ = self._get_warped_schedule(t, mode=mode)
        val = dot_alpha if deriv else alpha
        
        # 1. Permute from [B, 3, H, W] to [B, H, W, 3] for matrix multiplication
        x_perm = x.permute(0, 2, 3, 1)
        
        # 2. Apply Color Matrix (Broadcasts over B, H, W)
        x_yuv = torch.matmul(x_perm, self.rgb_to_yuv.T)
        
        # 3. Construct Coefficients Tensor [B, 1, 1, 3]
        B = x.shape[0]
        coeffs = torch.zeros((B, 1, 1, 3), device=self.device)
        
        if not deriv:
            # Preserve Luma (Y) channel at index 0
            coeffs[..., 0] = 1.0
            
        # Scale Chroma (UV) channels at indices 1, 2
        # Broadcast val [B, 1, 1, 1] into [B, 1, 1, 2]
        coeffs[..., 1:] = val.view(B, 1, 1, 1)
        
        # 4. Apply Mask (Broadcasts [B, 1, 1, 3] over [B, H, W, 3])
        x_yuv_masked = x_yuv * coeffs
        
        # 5. Convert back to RGB and permute to [B, 3, H, W]
        return torch.matmul(x_yuv_masked, self.yuv_to_rgb.T).permute(0, 3, 1, 2)


    def op_spatial(self, x, t, deriv=False, mode='linear'):
        """Delayed, Patch-based Spatial Masking """
        # Default Warping: Linear
        alpha, _, dot_alpha, _, _ = self._get_warped_schedule(t, mode=mode)
        
        # Logic:
        # In "Masked" regions (self.spatial_mask == 1), signal scales by alpha(t).
        # In "Visible" regions (self.spatial_mask == 0), signal stays at 1.0 (or scales by alpha_clean if desired).
        # Report implies "occlusion", so masked regions lose signal.
        
        # Invert mask for multiplication: 1=Keep, 0=Kill
        # But we want smooth degradation.
        # Visible regions: Scale = 1.0 (No degradation)
        # Masked regions: Scale = alpha (Degrades to 0/Noise)
        
        patch_map = (1 - self.spatial_mask) * 1.0 + self.spatial_mask * alpha
        
        if deriv:
            # Derivative of 1.0 is 0
            # Derivative of alpha is dot_alpha
            patch_map_deriv = self.spatial_mask * dot_alpha
            return x * patch_map_deriv
            
        return x * patch_map


    ###! check this code and the variance and present the statistics 

    def compute_target(self, x0, t):
        """
        Computes x_t and u_t with warped schedules.
        
        Args:
            x0: Clean image batch [B, C, H, W]
            t: Time batch [B, 1, 1, 1]
            modes: List of scheduling modes (must match length of operators)
            operators: Optional list of operator functions to apply. 
                       Defaults to [op_freq, op_color, op_spatial].
        """

        len_op = len(self.operators)
        
        # For noise scaling S_k, we usually match the subspace schedule
        # to ensure signal+noise variance = 1 (VP).
        # We need specific sigmas for each subspace based on their warping.
        
        sigmas = []
        dot_sigmas = []
        
        for m in self.modes:
            _, s, _, ds, _ = self._get_warped_schedule(t, mode=m)
            sigmas.append(s)
            dot_sigmas.append(ds)
            
        epsilons = [torch.randn_like(x0) for _ in range(len_op)]
        
        ##* Signal Path (mu_t)
        states = [x0]
        curr = x0
        for i, op in enumerate(self.operators):
            curr = op(curr, t, deriv=False, mode=self.modes[i])
            states.append(curr)
        mu_K = states[-1]
        
        dot_mu_K = torch.zeros_like(x0)
        for j in range(len_op):
            term = self.operators[j](states[j], t, deriv=True, mode=self.modes[j])
            for k in range(j+1, len_op):
                term = self.operators[k](term, t, deriv=False, mode=self.modes[k])
            dot_mu_K += term

        ##* Noise Path (eta_t)
        eta_t = torch.zeros_like(x0)
        dot_eta_t = torch.zeros_like(x0)

        for k in range(len_op):
            sigma = sigmas[k]
            dot_sigma = dot_sigmas[k]
            eps = epsilons[k]
            
            # A. State Contribution
            noise_contribution = eps * sigma
            for m in range(k+1, len_op):
                noise_contribution = self.operators[m](noise_contribution, t, deriv=False, mode=self.modes[m])
            eta_t += noise_contribution
            
            # B. Velocity Contribution
            # B1. S_k derivative
            vel_term_1 = eps * dot_sigma
            for m in range(k+1, len_op):
                vel_term_1 = self.operators[m](vel_term_1, t, deriv=False, mode=self.modes[m])
                
            # B2. Operator derivatives
            vel_term_2 = torch.zeros_like(x0)
            base_vec = eps * sigma
            n_states = [base_vec]
            curr_n = base_vec
            remaining_indices = range(k+1, len_op)
            for m in remaining_indices:
                curr_n = self.operators[m](curr_n, t, deriv=False, mode=self.modes[m])
                n_states.append(curr_n)
            
            for idx, m in enumerate(remaining_indices):
                d_term = self.operators[m](n_states[idx], t, deriv=True, mode=self.modes[m])
                for p in range(m+1, len_op):
                    d_term = self.operators[p](d_term, t, deriv=False, mode=self.modes[p])
                vel_term_2 += d_term
                
            dot_eta_t += (vel_term_1 + vel_term_2)

        return mu_K + eta_t, (dot_mu_K + dot_eta_t)/math.sqrt(2 * len_op)
    



class DecoupledOTCFM(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), device='cuda', r_low=0.2):
        super().__init__()
        self.device = device
        self.C, self.H, self.W = img_shape
        
        # We assume FastDCT2D is defined as in your previous file
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
        # Center the frequencies for the radial mask
        self.freq_dist = torch.sqrt((x / self.W)**2 + (y / self.H)**2) / math.sqrt(2)
        # Binary Content Mask: P_c (low freq)
        self.freq_P_c = (self.freq_dist <= r_low).float().unsqueeze(0).unsqueeze(0)
        # Binary Style Mask: P_s (high freq)
        self.freq_P_s = 1.0 - self.freq_P_c
        
        # 3. Orthogonal Spatial Projections (Patch Basis)
        self.patch_size = 8
        h_patches, w_patches = self.H // self.patch_size, self.W // self.patch_size
        torch.manual_seed(42)
        mask_grid = (torch.rand((1, 1, h_patches, w_patches), device=device) > 0.4).float()
        self.spatial_mask = torch.nn.functional.interpolate(mask_grid, size=(self.H, self.W), mode='nearest')
        # Content: Visible regions. Style: Dropped regions.
        self.spatial_P_c = 1.0 - self.spatial_mask 
        self.spatial_P_s = self.spatial_mask


    def _get_curriculum_rates(self, t, k):
        """
        Computes the survival rates and their analytic time-derivatives.
        gamma(t) = 1 - t
        alpha(t) = (1 - t) * exp(-kt)
        """
        # Content Survival
        gamma = 1.0 - t
        dot_gamma = -torch.ones_like(t)
        
        # Style/Nuisance Survival (Early Death)
        exp_decay = torch.exp(-k * t)
        alpha = gamma * exp_decay
        dot_alpha = -exp_decay - k * gamma * exp_decay
        
        return gamma, alpha, dot_gamma, dot_alpha

    # --- Recursive Operators ---
    # To compute d/dt [M(t) x(t)] efficiently, each operator accepts 
    # both the state `x` and its accumulating derivative `dx`.
    # It applies the product rule: new_dx = M_dot * x + M * dx

    def op_freq(self, x, dx, t, k_freq):
        gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, k_freq)
        
        # Construct exact scaling matrices based on P_c and P_s
        scale_x  = gamma * self.freq_P_c + alpha * self.freq_P_s
        scale_dx = dot_gamma * self.freq_P_c + dot_alpha * self.freq_P_s
        
        # Forward DCT
        X_freq = self.dct(x)
        dX_freq = self.dct(dx)
        
        # Apply Structural Drift
        new_X_freq = X_freq * scale_x
        # Apply Analytic Product Rule for the Vector Field
        new_dX_freq = X_freq * scale_dx + dX_freq * scale_x
        
        return self.dct.inverse(new_X_freq), self.dct.inverse(new_dX_freq)


    def op_color(self, x, dx, t, k_color):
        gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, k_color)
        
        x_perm = x.permute(0, 2, 3, 1)
        dx_perm = dx.permute(0, 2, 3, 1)
        
        # Change to YUV basis
        yuv_x = torch.matmul(x_perm, self.rgb_to_yuv.T)
        yuv_dx = torch.matmul(dx_perm, self.rgb_to_yuv.T)
        
        B = x.shape[0]
        # P_c = Luma (idx 0), P_s = Chroma (idx 1, 2)
        scale_x = torch.cat([gamma, alpha, alpha], dim=-1).view(B, 1, 1, 3)
        scale_dx = torch.cat([dot_gamma, dot_alpha, dot_alpha], dim=-1).view(B, 1, 1, 3)
        
        # Apply Structural Drift
        new_yuv_x = yuv_x * scale_x
        # Apply Analytic Product Rule
        new_yuv_dx = yuv_x * scale_dx + yuv_dx * scale_x
        
        new_x = torch.matmul(new_yuv_x, self.yuv_to_rgb.T).permute(0, 3, 1, 2)
        new_dx = torch.matmul(new_yuv_dx, self.yuv_to_rgb.T).permute(0, 3, 1, 2)
        
        return new_x, new_dx


    def op_spatial(self, x, dx, t, k_spatial):
        gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, k_spatial)
        
        scale_x = gamma * self.spatial_P_c + alpha * self.spatial_P_s
        scale_dx = dot_gamma * self.spatial_P_c + dot_alpha * self.spatial_P_s
        
        new_x = x * scale_x
        new_dx = x * scale_dx + dx * scale_x
        
        return new_x, new_dx


# ###* posssible improvement:
#     def op_illumination(self, x, dx, t, k_illum):
#         gamma, alpha, dot_gamma, dot_alpha = self._get_curriculum_rates(t, k_illum)
        
#         # Compute P_s (The Global Mean)
#         mean_x = torch.mean(x, dim=(2, 3), keepdim=True)
#         mean_dx = torch.mean(dx, dim=(2, 3), keepdim=True)
        
#         # Compute P_c (The AC Contrast)
#         contrast_x = x - mean_x
#         contrast_dx = dx - mean_dx
        
#         # Apply exactly as derived
#         new_x = gamma * contrast_x + alpha * mean_x
#         new_dx = dot_gamma * contrast_x + gamma * contrast_dx + dot_alpha * mean_x + alpha * mean_dx
        
#         return new_x, new_dx


    # --- The Core CFM Computation ---

    def compute_target(self, x0, t, k_rates={'freq': 5.0, 'color': 3.0, 'spatial': 2.0}):
        """
        Computes the decoupled OT-CFM path and its exact, simulation-free target vector field.
        """
        # 1. Initialize the recursive states
        s_curr = x0
        ds_curr = torch.zeros_like(x0) # Derivative of clean data is 0
        
        # 2. Apply sequential non-commutative operators
        # We sequentially build M_t(x0) and \dot{M}_t(x0)
        s_curr, ds_curr = self.op_freq(s_curr, ds_curr, t, k_rates['freq'])
        s_curr, ds_curr = self.op_color(s_curr, ds_curr, t, k_rates['color'])
        s_curr, ds_curr = self.op_spatial(s_curr, ds_curr, t, k_rates['spatial'])
        
        M_t_x0 = s_curr
        dot_M_t_x0 = ds_curr
        
        # 3. Inject single isotropic noise variable
        # The explicit noise variable is uniquely determined by the endpoints.
        eps = torch.randn_like(x0)
        
        # 4. Construct Final Path and Target Vector Field
        # x_t = M_t x_0 + t * eps
        x_t = M_t_x0 + t * eps
        
        # u_t = \dot{M}_t x_0 + eps
        u_t = dot_M_t_x0 + eps
        
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