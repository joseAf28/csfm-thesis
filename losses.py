# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
import math
import random
import torch.nn.functional as F
import logging

from models import utils as mutils


def get_optimizer(config, model_module):
    """Returns a PyTorch optimizer with Two-Time-Scale Update Rules (TTUR)."""
    
    # 1. Separate the parameters by looking at their names
    encoder_params = []
    unet_params = []
    
    for name, param in model_module.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            unet_params.append(param)
            
    # 2. Define the Differential Learning Rates
    unet_lr = config.optim.lr
    encoder_lr_multiplier = getattr(config.optim, 'encoder_lr_mult', 0.1)
    encoder_lr = unet_lr * encoder_lr_multiplier
    
    # 3. Create the Parameter Groups (ADD 'initial_lr' HERE)
    param_groups = [
        {'params': unet_params, 'lr': unet_lr, 'initial_lr': unet_lr},
        {'params': encoder_params, 'lr': encoder_lr, 'initial_lr': encoder_lr}
    ]
    
    print(f"param_groups: UNet LR={param_groups[0]['lr']}, Encoder LR={param_groups[1]['lr']}")
    
    # 4. Initialize the Optimizer
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            param_groups, 
            betas=(config.optim.beta1, 0.999), 
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')
        
    return optimizer



def optimization_manager(config):
    """Returns an optimize_fn based on `config` for warmup and clipping."""
    def optimize_fn(optimizer, params, step, 
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        
        if warmup > 0:
            for g in optimizer.param_groups:
                # Fetch the specific base LR for this group
                base_lr = g.get('initial_lr', config.optim.lr)
                # Scale the specific base LR by the warmup factor
                g['lr'] = base_lr * np.minimum(step / warmup, 1.0)
                
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            
        optimizer.step()
        
    return optimize_fn



##* Baseline Methods
def autoencoder_loss_fn(model, batch, regularizer, config, reduce_op):
    """
    Standard AE: Reconstruct x_0 directly from z using the main U-Net.
    The U-Net downsampling path is starved with zeros to force all info through z.
    """
    B = batch.shape[0]
    device = batch.device
    
    x_dummy = torch.zeros_like(batch)
    t_dummy = torch.zeros(B, device=device)
    
    # Forward pass: 
    # x=x_dummy (U-Net downsampling sees nothing but zeros)
    # x0=batch (WideResNet sees the real image to create z)
    # mode='ae' (Severs skip connections just to be perfectly safe)
    model_result = model(x=x_dummy, time_cond=t_dummy, x0=batch, mode='ae')
    
    x_reconstructed = model_result['output']
    z = model_result['latent']
    
    losses_task = torch.square(x_reconstructed - batch) 
    losses_task = reduce_op(losses_task.reshape(B, -1), dim=-1)
    
    raw_reg = torch.tensor(0.0, device=device)
    losses_reg = torch.tensor(0.0, device=device)
    
    if config.training.lambda_reg > 0.0:
        raw_reg = regularizer(z)
        losses_reg = config.training.lambda_reg * raw_reg
    
    task_loss = torch.mean(losses_task)
    total_loss = task_loss + losses_reg
    
    # Return the expected 3-tuple
    return total_loss, task_loss, raw_reg



def cdae_loss_fn(model, batch, flow_engine, regularizer, config, reduce_op):
    """
    Conditional Denoising AE supporting both OT and CSFM degradation paths.
    Denoises x_t directly back to x_0 in one step.
    """
    B = batch.shape[0]
    device = batch.device
    
    t = torch.rand(B, device=device).view(-1, 1, 1, 1)
    deg_path = getattr(config.training, 'cdae_degradation', 'ot').lower()
    
    if deg_path == 'csfm':
        x_t, _ = flow_engine.compute_target(batch, t)
    else:
        x_1 = torch.randn_like(batch)
        x_t = (1 - t) * batch + t * x_1 
    
    model_result = model(x_t, t.flatten(), x0=batch, mode='cdae')
    v_pred = model_result['output'] 
    z = model_result['latent']
    
    losses_task = torch.square(v_pred - batch)
    losses_task = reduce_op(losses_task.reshape(B, -1), dim=-1)
    
    raw_reg = torch.tensor(0.0, device=device)
    losses_reg = torch.tensor(0.0, device=device)
    
    if config.training.lambda_reg > 0.0:
        raw_reg = regularizer(z)
        losses_reg = config.training.lambda_reg * raw_reg
    
    task_loss = torch.mean(losses_task)
    total_loss = task_loss + losses_reg
    
    return total_loss, task_loss, raw_reg


###* JEA loss 
def lejea_loss_fn(model, batch, regularizer, config, reduce_op):
    """
    Dynamically routes the JEA objective based on architectural constraints.
    """
    x1, x2 = batch
    B = x1.shape[0]
    device = x1.device
    
    x_dummy = torch.zeros_like(x1)
    t_dummy = torch.ones(B, device=device)
    
    # 1. Extract representations
    z1 = model(x_dummy, t_dummy, x0=x1, mode='lejea')['latent']
    z2 = model(x_dummy, t_dummy, x0=x2, mode='lejea')['latent']
    
    # Check if the architecture is providing BatchNorm protection
    use_projector = getattr(config.model, 'use_jea_projector', False)
    
    if use_projector:
        ##* PATH A: Pure Euclidean
        # The MLP Projector contains BatchNorm, which mathematically bounds he variance. 
        diff_sq = torch.sum(torch.square(z1 - z2), dim=-1)
    else:
        ##* PATH B: Hybrid Spherical-Euclidean
        # Bare ResNet is used. Without BatchNorm, pure Euclidean collapses 
        # to the origin. We MUST map the invariance task to the unit hypersphere.
        z1_norm = torch.nn.functional.normalize(z1, dim=-1)
        z2_norm = torch.nn.functional.normalize(z2, dim=-1)
        diff_sq = torch.sum(torch.square(z1_norm - z2_norm), dim=-1)
        
    # 2. Compute semantic task loss (Angle alignment or Euclidean proximity)
    task_loss = 0.25 * torch.mean(diff_sq) 
    
    # 3. Apply Regularizer to RAW representations (Builds the Gaussian prior)
    raw_reg = 0.5 * (regularizer(z1) + regularizer(z2))
    
    # 4. Strict Convex Combination (Balances the gradient scales)
    lam = getattr(config.training, 'lambda_reg', 0.05) 
    total_loss = lam * raw_reg + (1.0 - lam) * task_loss
    
    return total_loss, task_loss, raw_reg




def flow_matching_loss_fn(model, batch, flow_engine, regularizer, config, reduce_op):
    """
    Composite Subspace Flow Matching (CSFM) Objective:
    Regress the continuous vector field u_t with Sparsely Supervised Diffusion (SSD) 
    trace decoupling and Asymmetric Gradient Routing to prevent encoder gradient starvation.
    """
    
    B = batch.shape[0]
    device = batch.device

    ##* time sampling
    time_mode = getattr(config.training, 'time_sampling_mode', 'uniform')
    tc = getattr(config.training, 't0', 1e-8)
    
    if time_mode == 'uniform':
        t = torch.rand(B, device=device)
    elif time_mode == 'beta':
        # Beta distribution bounds to [0,1]
        a = getattr(config.training, 'time_beta_a', 2.0)
        b = getattr(config.training, 'time_beta_b', 2.0)
        t = torch.distributions.beta.Beta(a, b).sample((B,)).to(device)
    elif time_mode == 'piecewise_linear_uniform':
        
        # Calculate PDF constants and CDF threshold
        c = 1.0 / (1.0 - (tc / 2.0))
        a = c / tc
        Uc = 0.5 * a * (tc ** 2)
        # Inverse Transform Sampling
        U = torch.rand(B, device=device)
        t = torch.where(
            U <= Uc,
            torch.sqrt((2.0 * U) / a),            # Invert the linear ramp
            tc + ((U - Uc) / c)                   # Invert the uniform flat section
        )
        
    # Broadcast time tensor for image dimensions (B, 1, 1, 1)
    t = t.view(-1, 1, 1, 1)
    
    ##* time-dependent loss weighting
    weight_mode = getattr(config.training, 'loss_weighting_mode', 'none')
    
    if weight_mode == 'none':
        loss_weight = 1.0
    elif weight_mode == 'variance_inverse':
        tc = getattr(config.training, 't0', 0.1)
        loss_weight = torch.where(
            t <= tc,
            t / tc,
            torch.tensor(1.0, device=device)
        )
    
    ##* compute targte flow
    x_t, u_t = flow_engine.compute_target(batch, t)
    
    ##* asymmteric gradient routing
    # Step 4a: Extract raw latent from the encoder using the 'jea' mode.
    # We pass x=None and time_cond=None because the encoder only processes x0.
    model_enc_result = model(x=None, time_cond=None, x0=batch, mode='flow_enc')
    z_raw = model_enc_result['latent']
    
    # Step 4b: Vectorized Temporal Masking
    # We stop gradients to the encoder when t < t_c to prevent memorization 
    # of high-frequency style details near the clean data boundary.
    
    # Create mask: 1.0 if t >= t_c (allow gradient), 0.0 if t < t_c (stop gradient)
    routing_mask = (t.flatten() >= tc).float().view(B, -1)
    
    # Detach a copy of the latent vector from the computation graph
    z_detached = z_raw.detach()
    
    # Route the latent dynamically per-sample based on its timestep
    z_routed = z_raw * routing_mask + z_detached * (1.0 - routing_mask)
    
    # Step 4c: Predict vector field using the ROUTED latent
    # We pass z_routed into the newly added z_cond argument of your NCSN++ model.
    model_result = model(x_t, t.flatten(), x0=batch, z_cond=z_routed, mode='flow')
    v_pred = model_result['output']
    
    
    ##* ssd masking
    eta = getattr(config.training, 'ssd_eta', 0.0)
    
    # Calculate raw squared errors
    squared_errors = torch.square(v_pred - u_t) * loss_weight
    
    if eta > 0.0:
        # Generate spatial mask: 1 = keep (1-eta prob), 0 = drop (eta prob)
        mask = (torch.rand(B, 1, batch.shape[2], batch.shape[3], device=device) > eta).float()
        
        # Scale up surviving errors to match unmasked expected gradient energy
        channels = batch.shape[1]
        total_elements = batch.shape[2] * batch.shape[3] * channels
        num_active_elements = (torch.sum(mask.reshape(B, -1), dim=-1) * channels).clamp(min=1.0)
        
        scale_factor = (total_elements / num_active_elements).view(B, 1, 1, 1)
        squared_errors = squared_errors * mask * scale_factor
    
    ##* flow loss
    losses_flow = reduce_op(squared_errors.reshape(B, -1), dim=-1)
    task_loss = torch.mean(losses_flow)
    
    ##* dimensional collapse regularization
    #! applied to raw latent,not the routed one - encoder must optimize its latent manifold over the entire batch, even if the U-net flow ignreos 
    raw_reg_loss = torch.tensor(0.0, device=device)
    losses_reg = torch.tensor(0.0, device=device)
    
    if getattr(config.training, 'lambda_reg', 0.0) > 0.0:
        raw_reg_loss = regularizer(z_raw)
        losses_reg = config.training.lambda_reg * raw_reg_loss

    total_loss = task_loss + losses_reg

    return total_loss, task_loss, raw_reg_loss




def get_loss_fn(flow_engine, regularizer, train, reduce_mean=True, config=None):
    """
    Dispatcher that returns the correct loss function based on the config.
    """
    loss_type = getattr(config.training, 'loss_type', 'csfm').lower()
    
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def step_loss_fn(model, batch):
        if loss_type == 'ae':
            return autoencoder_loss_fn(model, batch, regularizer, config, reduce_op)
            
        elif loss_type == 'cdae':
            return cdae_loss_fn(model, batch, flow_engine, regularizer, config, reduce_op)
            
        elif loss_type in ['csfm', 'ot']:
            return flow_matching_loss_fn(model, batch, flow_engine, regularizer, config, reduce_op)
    
        elif loss_type == 'lejea':
            return lejea_loss_fn(model, batch, regularizer, config, reduce_op)
    
        else:
            raise ValueError(f"Unknown loss_type in config: {loss_type}")

    return step_loss_fn



def get_step_fn(flow_engine, regularizer, train, optimize_fn=None, reduce_mean=True, config=None):
    """
    Universal step function for AE, CDAE, OT, and CSFM.
    Performs one optimization step (Forward + Backward + Update).
    """
    
    loss_fn = get_loss_fn(
        flow_engine=flow_engine,
        regularizer=regularizer,
        train=train,
        reduce_mean=reduce_mean,
        config=config
    )

    def step_fn(state, batch):
        model = state['model']
        
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            
            # The PyTorch graph is built dynamically here based on the loss_fn routed above
            ##* unpack tuple
            total_loss, task_loss, raw_reg = loss_fn(model, batch)
            total_loss.backward()
            
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                total_loss, task_loss, raw_reg = loss_fn(model, batch)
                ema.restore(model.parameters())

        return {
            'total': total_loss.detach(),
            'task': task_loss.detach(),
            "raw_reg": raw_reg.detach()
        }

    return step_fn
