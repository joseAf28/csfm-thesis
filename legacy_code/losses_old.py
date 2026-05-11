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
from models import utils as mutils
from sde_lib import VESDE, VPSDE

import logging


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn



# --- 1. THE LOSS CALCULATOR (Math) ---
def get_flow_loss_fn(flow_engine, regularizer, train, reduce_mean=True, config=None):
    """
    Returns a function that computes the loss for a single batch.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        # 1. Sample Time t
        t = torch.rand(batch.shape[0], device=batch.device).view(-1, 1, 1, 1)

        # 2. Compute Target (Physics)
        x_t, u_t = flow_engine.compute_target(batch, t)

        # 3. Model Prediction
        # Pass t=None to encoder if we want it to be time-independent and x_0 for the encoder 
        model_result = model(x_t, t.flatten(), x0=batch)
        
        v_pred = model_result['output']
        z = model_result['latent']

        # 4. Matching Loss
        losses_flow = torch.square(v_pred - u_t)
        losses_flow = reduce_op(losses_flow.reshape(losses_flow.shape[0], -1), dim=-1)

        # 5. Regularization
        losses_reg = torch.tensor(0.0, device=batch.device)
        if config.training.lambda_reg > 0.0:
            reg_val = regularizer(z)
            losses_reg = config.training.lambda_reg * reg_val

        return torch.mean(losses_flow) + losses_reg

    return loss_fn


# --- 2. THE OPTIMIZATION STEP (Mechanics) ---
def get_flow_step_fn(flow_engine, regularizer, train, optimize_fn=None, reduce_mean=True, config=None):
    """
    Returns a function that performs one optimization step (Forward + Backward + Update).
    """
    loss_fn = get_flow_loss_fn(
        flow_engine=flow_engine,
        regularizer=regularizer,
        train=train,
        reduce_mean=reduce_mean,
        config=config
    )

    def step_fn(state, batch):
        model = state['model']
        
        # --- TRAINING ---
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            
            loss = loss_fn(model, batch)
            loss.backward()
            
            # Apply Grad Clipping & Warmup
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            
            state['step'] += 1
            state['ema'].update(model.parameters())

        # --- EVALUATION ---
        else:
            with torch.no_grad():
                # Use EMA weights for validation
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                
                loss = loss_fn(model, batch)
                
                ema.restore(model.parameters())

        return loss

    return step_fn



def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False,
                config=None):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting, config=config)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            use_constrained_architecture = getattr(config.model, 'constrained_architecture', False)
            if use_constrained_architecture:
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())
            else:
                with torch.no_grad():
                    ema = state['ema']
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    loss = loss_fn(model, batch)
                    ema.restore(model.parameters())

        return loss

    return step_fn




def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, config=None):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        x_t = mean + std[:, None, None, None] * z

        x0_input = batch
        loss_weight = 1.0
        apply_mixup = getattr(config.training, 'apply_mixup', False)
        if apply_mixup:
            alpha = 1.0
            lam = torch.from_numpy(np.random.beta(alpha, alpha, batch.shape[0]).astype(np.float32)).to(batch.device)
            index = torch.randperm(batch.shape[0], device=batch.device)
            x0_input = lam[:, None, None, None] * batch + (1. - lam[:, None, None, None]) * batch[index, :]
            loss_weight = 2 * lam

        model_result = score_fn(x_t, t, x0=x0_input)
        score = model_result['output']

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        losses *= loss_weight

        losses_reg = 0 * losses
        if config.training.lambda_z > 0.0:
            if config.training.probabilistic_encoder:
                mu, log_var = model_result['z_mean'], model_result['z_logvar']
                kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1)
                losses_reg += config.training.lambda_z * kl
                if not train:
                    logging.info(f'avg abs mean {mu.absolute().mean()}, avg std {(0.5 * log_var).exp().mean()}')
            else:
                losses_reg += config.training.lambda_z * torch.sum(torch.abs(model_result['latent']), dim=-1)

        if config.training.lambda_reconstr != 0.0:
            recon_method = getattr(config.training, 'recon', 'l2')
            if recon_method == 'bce':
                losses_reconstr = torch.nn.BCEWithLogitsLoss(reduction='none')(model_result['reconstr'], batch)
            elif recon_method == 'l2':
                losses_reconstr = torch.nn.MSELoss(reduction='none')(model_result['reconstr'], batch)
            else:
                print("Not Supported")
                exit()

            losses_reconstr = reduce_op(losses_reconstr.reshape(losses_reconstr.shape[0], -1), dim=-1)
            if config.training.lambda_reconstr < 0.0:
                losses = losses * 0 + losses_reconstr
            else:
                losses += config.training.lambda_reconstr * losses_reconstr

        losses += losses_reg
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                         sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


