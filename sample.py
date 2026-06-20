import os
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import logging
import argparse
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import gc

from torchmetrics.image.fid import FrechetInceptionDistance

from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
from run_lib import get_dataloaders

##* DATASET STATISTICS FOR UN-NORMALIZATION
channel_stats = {
    'cifar10': dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'cifar100': dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    'mini_imgnet': dict(mean=[120.3958/255.0, 115.5936/255.0, 104.5401/255.0], 
                        std=[70.6818/255.0, 68.2763/255.0, 72.5450/255.0])
}

def unnormalize_to_uint8(tensor, dataset_name):
    stats = channel_stats.get(dataset_name, channel_stats['cifar10'])
    mean = torch.tensor(stats['mean']).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(stats['std']).view(1, 3, 1, 1).to(tensor.device)
    
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return (tensor * 255).to(torch.uint8)

##* MATHEMATICAL UTILITIES
def get_time_schedule(N_steps, rho=1.0, device='cpu'):
    j = torch.arange(N_steps + 1, dtype=torch.float32, device=device)
    t_seq = (1.0 - j / N_steps) ** rho
    return t_seq


def slerp(z1, z2, alpha):
    """
    Spherical Linear Interpolation (Slerp) for high-dimensional latents.
    z1, z2: Tensors of shape [B, D] or [B, C, H, W]
    alpha: Float interpolation parameter (can be outside [0,1] for extrapolation)
    """
    # Flatten spatial dimensions if necessary
    z1_flat = z1.view(z1.shape[0], -1)
    z2_flat = z2.view(z2.shape[0], -1)
    
    # Normalize to unit vectors
    z1_norm = z1_flat / (torch.norm(z1_flat, dim=-1, keepdim=True) + 1e-8)
    z2_norm = z2_flat / (torch.norm(z2_flat, dim=-1, keepdim=True) + 1e-8)
    
    # Compute the angle (omega) between the vectors
    dot_product = torch.clamp((z1_norm * z2_norm).sum(dim=-1, keepdim=True), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    
    # If vectors are parallel (sin_omega ~ 0), fallback to standard lerp to avoid NaNs
    fallback_lerp = (1.0 - alpha) * z1_flat + alpha * z2_flat
    
    # Slerp formula
    slerp_res = torch.sin((1.0 - alpha) * omega) / sin_omega * z1_flat + \
                torch.sin(alpha * omega) / sin_omega * z2_flat
                
    res = torch.where(sin_omega < 1e-5, fallback_lerp, slerp_res)
    
    return res.view(z1.shape)


class DummyEncoder(torch.nn.Module):
    def __init__(self, z_val):
        super().__init__()
        self.z_val = z_val
    def forward(self, x=None, t=None):
        return None, None, self.z_val


##* ADVANCED ODE SOLVER (GPU OPTIMIZED)
def ode_solve(model, z, device, img_shape=(3, 32, 32), N_steps=50, method='heun', rho=1.0, track_trajectory=False, x_init=None):
    model.eval()
    B = z.shape[0]
    
    if x_init is not None:
        x = x_init.clone().to(device)
    else:
        x = torch.randn((B, *img_shape), device=device)
        
    t_seq = get_time_schedule(N_steps, rho=rho, device=device)

    original_encoder = getattr(model, 'encoder', None)
    if original_encoder is not None:
        model.encoder = DummyEncoder(z)
    
    trajectory = []
    velocities = []

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i in range(N_steps):
            if track_trajectory and (i % max(1, N_steps // 10) == 0):
                trajectory.append(x.detach().cpu()) # Safe if occasional, but consider keeping on GPU if memory allows

            t_curr = t_seq[i]
            t_next = t_seq[i+1]
            dt = t_curr - t_next
            
            # OPTIMIZATION: Zero-overhead tensor creation
            t_curr_tensor = torch.full((B,), t_curr, device=device, dtype=x.dtype)
            
            v_curr = model(x, t_curr_tensor, x0=x, mode='flow')['output']
            
            # OPTIMIZATION: Keep velocity tracking entirely on the GPU to prevent CUDA syncing
            velocities.append(v_curr.detach().view(B, -1).unsqueeze(1)) 
            
            if method == 'euler' or i == N_steps - 1:
                x = x - v_curr * dt
            elif method == 'heun':
                x_hat = x - v_curr * dt
                t_next_tensor = torch.full((B,), t_next, device=device, dtype=x.dtype)
                
                v_next = model(x_hat, t_next_tensor, x0=x_hat, mode='flow')['output']
                x = x - (dt / 2.0) * (v_curr + v_next)
            else:
                raise ValueError(f"Unknown integration method: {method}")
            
    if original_encoder is not None:
        model.encoder = original_encoder
        
    if track_trajectory:
        trajectory.append(x.detach().cpu())

    # OPTIMIZATION: Compute the variance on the GPU, then pull only the final float to CPU
    velocities_tensor = torch.cat(velocities, dim=1) 
    velocity_variance = torch.var(velocities_tensor, dim=1).mean().item()

    return x, trajectory, velocity_variance



def run_sampling_tests(config, workdir, ckpt_path, out_dir, method, steps, rho, eval_loader, device, condition_mode='unconditional', target_fid_samples=1000):
    """
    Executes the three generative tests required for Thesis Section 4.4,
    supporting both Conditional (Reconstruction) and Unconditional (Prior) generation.
    """

    img_shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    loss_type = getattr(config.training, 'loss_type', 'csfm').lower()
    
    model = mutils.create_model(config).float().to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

    checkpoint_dir = os.path.join(workdir, "checkpoints_" + config.training.experiment_name)
    enc_ckpt_path = None
    
    if ckpt_path and os.path.exists(ckpt_path):
        latest_ckpt = ckpt_path
        try:
            step_num = latest_ckpt.split('_')[-1].split('.')[0]
            enc_ckpt_path = latest_ckpt.replace('checkpoints_', 'checkpointsenc_').replace(f'checkpoint_{step_num}.pth', f'encoder_state_{step_num}.pth')
        except Exception:
            pass
    else:
        ckpt_pattern = os.path.join(checkpoint_dir, 'checkpoint_*.pth')
        ckpt_files = glob.glob(ckpt_pattern)
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        state = restore_checkpoint(latest_ckpt, dict(step=0), device)
        save_step_idx = int(state.get('step', 0)) // config.training.snapshot_freq
        enc_ckpt_path = os.path.join(workdir, "checkpointsenc_" + config.training.experiment_name, f'encoder_state_{save_step_idx}.pth')

    logging.info(f"Restoring main model from: {latest_ckpt}")
    loaded_state = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(loaded_state['model'], strict=False)
    ema.load_state_dict(loaded_state['ema'])
    
    if getattr(config.training, 'include_encoder', False) and enc_ckpt_path and os.path.exists(enc_ckpt_path):
        model.encoder.load_state_dict(torch.load(enc_ckpt_path, map_location=device, weights_only=False))

    ema.copy_to(model.parameters())
    model.eval()
    
    # 2. Determine which modes to execute
    if condition_mode.lower() == 'all':
        modes_to_run = ['unconditional', 'conditional']
    else:
        modes_to_run = [condition_mode.lower()]

    logging.info(f"--- Starting Sampling {loss_type.upper()} | Modes: {modes_to_run} ---")
    
    for current_mode in modes_to_run:
        # Adjust output directory to reflect the condition mode so you don't overwrite files
        config_folder_name = f"{method}_N{steps}_rho{rho}"
        current_out_dir = os.path.join(out_dir, current_mode, config_folder_name)
        os.makedirs(current_out_dir, exist_ok=True)
        
        logging.info(f"\n>>> Executing Mode: {current_mode.upper()} <<<")
        logging.info(f"Saving to: {current_out_dir}")
        
        # Small-Sample FID (N=1000) & Qualitative Grids
        logging.info(f"Executing FID: Generating {target_fid_samples} samples via {method.upper()} solver...")
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        
        generated_samples = []
        real_samples = []
        
        pbar_fid = tqdm(total=target_fid_samples, desc=f"FID Generation ({current_mode})", unit="img")
        
        with torch.inference_mode():
            for batch, _ in eval_loader:
                batch = batch.to(device, non_blocking=True)
                B = batch.shape[0]
                # Prevent generating more than necessary on the final batch
                if sum([b.shape[0] for b in generated_samples]) + B > target_fid_samples:
                    B = target_fid_samples - sum([b.shape[0] for b in generated_samples])
                    batch = batch[:B]
                
                real_samples.append(batch.cpu())
                
                # Boundary Condition Setup (t=1)
                if current_mode == 'conditional':
                    z = model.encoder(batch)[2]
                elif current_mode == 'unconditional':
                    latent_dim = int(64 * getattr(config.model, 'widen_factor', 2))
                    z = torch.randn((batch.shape[0], latent_dim), device=device)
                else:
                    raise ValueError(f"Unknown condition mode: {current_mode}")
                
                # Integrate Flow
                gen_batch, _, _ = ode_solve(model, z, device, img_shape, N_steps=steps, method=method, rho=rho, track_trajectory=False)
                generated_samples.append(gen_batch.cpu())
                
                pbar_fid.update(B)
                if sum([b.shape[0] for b in generated_samples]) >= target_fid_samples:
                    break
                
        pbar_fid.close()
        # Stack and truncate to exactly N=1000
        all_gen = torch.cat(generated_samples, dim=0)
        all_real = torch.cat(real_samples, dim=0)
        
        # Save Uncurated Grid
        grid_path = os.path.join(current_out_dir, f"uncurated_grid_{loss_type}_{current_mode}.png")
        vutils.save_image(all_gen[:64], grid_path, nrow=8, normalize=True, value_range=(-1, 1))
        
        # Compute FID
        all_gen_uint8 = unnormalize_to_uint8(all_gen, config.data.dataset)
        all_real_uint8 = unnormalize_to_uint8(all_real, config.data.dataset)
        
        fid_metric.update(all_real_uint8.to(device), real=True)
        fid_metric.update(all_gen_uint8.to(device), real=False)
        relative_fid = fid_metric.compute().item()
        
        logging.info(f"Relative Small-Sample FID (N={target_fid_samples}): {relative_fid:.4f}")
        with open(os.path.join(current_out_dir, f"relative_fid_{loss_type}_{current_mode}.txt"), "w") as f:
            f.write(f"Relative FID (N={target_fid_samples}): {relative_fid:.4f}\n")
        
        # Latent Manifold Smoothness (Slerp Interpolation/Extrapolation)
        logging.info("Executing Slerp Latent Manifold Trajectories...")
        
        real_batch, _ = next(iter(eval_loader))
        real_batch = real_batch.to(device)
        
        num_pairs = 15
        alphas = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] # Extrapolates out of bounds
        
        ##* interpolation
        if current_mode == 'conditional':
            z_endpoints = model.encoder(real_batch[:num_pairs * 2])[2]
            
            real_anchors = torch.cat([real_batch[:num_pairs].cpu(), real_batch[num_pairs:num_pairs * 2].cpu()], dim=0)
            anchor_path = os.path.join(current_out_dir, f"slerp_real_anchors_{loss_type}_{current_mode}.png")
            # Saves 2 rows: Top row is Image A, Bottom row is Image B
            vutils.save_image(real_anchors, anchor_path, nrow=num_pairs, normalize=True, value_range=(-1, 1))
            logging.info(f"Saved explicit real target anchors for Slerp to {anchor_path}")

        elif current_mode == 'unconditional':
            latent_dim = int(64 * getattr(config.model, 'widen_factor', 2))
            z_endpoints = torch.randn((num_pairs * 2, latent_dim), device=device)
        
        z_A = z_endpoints[:num_pairs]
        z_B = z_endpoints[num_pairs:]
        
        all_interp_grids = []
        
        total_slerp_ops = num_pairs * len(alphas)
        pbar_slerp = tqdm(total=total_slerp_ops, desc=f"Slerp Grids ({current_mode})", unit="ode")
        
        with torch.inference_mode():
            for i in range(num_pairs):
                row_samples = []
                for alpha in alphas:
                    # Travel along the hypersphere surface
                    z_interp = slerp(z_A[i:i+1], z_B[i:i+1], alpha)
                    
                    # ODE Solve the interpolated latent back to pixels
                    gen_img, _, _ = ode_solve(model, z_interp, device, img_shape, N_steps=steps, method=method, rho=rho, track_trajectory=False)
                    row_samples.append(gen_img.cpu())
                    pbar_slerp.update(1)
                    
                row_tensor = torch.cat(row_samples, dim=0)
                all_interp_grids.append(row_tensor)
        pbar_slerp.close()
        
        # Stack all pairs vertically
        master_interp_tensor = torch.cat(all_interp_grids, dim=0)
        interp_path = os.path.join(current_out_dir, f"slerp_extrap_grid_{loss_type}_{current_mode}.png")
        
        vutils.save_image(master_interp_tensor, interp_path, nrow=len(alphas), normalize=True, value_range=(-1, 1))
        logging.info(f"Saved Slerp grid to {interp_path}")
        
        
        logging.info("Executing Trajectory Straightening (Truncation Error) Test...")
        B_trunc = min(real_batch.shape[0], 64) # Cap at 64 to avoid OOM on Baseline
        real_batch_trunc = real_batch[:B_trunc]
        
        if current_mode == 'conditional':
            z_eval = model.encoder(real_batch_trunc)[2]
        else:
            latent_dim = int(64 * getattr(config.model, 'widen_factor', 2))
            z_eval = torch.randn((B_trunc, latent_dim), device=device)
            
        x_shared = torch.randn((B_trunc, *img_shape), device=device)
        
        logging.info("Computing Baseline (Euler N=1000)...")
        with torch.inference_mode():
            # A simple TQDM to show baseline is running (since N=1000 takes time)
            for _ in tqdm([1], desc="Baseline N=1000 (Euler)", unit="calc"):
                x_baseline, _, var_baseline = ode_solve(
                    model, z_eval, device, img_shape, N_steps=1000, 
                    method='euler', rho=rho, x_init=x_shared
                )
            
        truncation_errors = {}
        eval_steps_list = [10, 20, 50, 100]
        
        # RELIABLE PROGRESS BAR: Track step sizes, display MSE live in postfix
        pbar_trunc = tqdm(eval_steps_list, desc=f"Truncation Err ({current_mode})", unit="step")
        
        for eval_steps in pbar_trunc:
            with torch.inference_mode():
                x_fast, _, var_fast = ode_solve(
                    model, z_eval, device, img_shape, N_steps=eval_steps, 
                    method='euler', rho=rho, x_init=x_shared
                )
            
            mse = torch.nn.functional.mse_loss(x_fast, x_baseline).item()
            truncation_errors[eval_steps] = mse
            
            # Dynamically update the progress bar text with the resulting MSE
            pbar_trunc.set_postfix({'MSE': f"{mse:.6f}"})
            logging.info(f"Truncation Error (N={eval_steps} vs N=1000): MSE = {mse:.6f} | Var = {var_fast:.4f}")
            
        with open(os.path.join(current_out_dir, f"truncation_error_{loss_type}_{current_mode}.txt"), "w") as f:
            for k, v in truncation_errors.items():
                f.write(f"Steps: {k}, MSE: {v:.6f}\n")
        
    logging.info("--- Section Sampling  Complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--method", type=str, default="heun")
    parser.add_argument("--steps", type=int, nargs='+', default=[20, 50, 100])
    parser.add_argument("--rho", type=float, nargs='+', default=[1.0, 3.0])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fid_samples", type=int, default=1000)
    args = parser.parse_args()
    pass