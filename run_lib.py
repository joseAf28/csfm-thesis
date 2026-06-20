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

# pylint: skip-file
"""Training and evaluation for  generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import logging
from models import ddpm, ncsnv2, ncsnpp
import losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage

import datasets
from absl import flags

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as T

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from collections import defaultdict
import matplotlib.pyplot as plt

from config import datasets as datasets_new
from config import cli
from config.augmentations import RandAugment
from torchvision.transforms import transforms
# import helpers
import torchvision

from functools import partial
import pandas as pd
import PIL
import glob

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, io
from torchvision.datasets.utils import verify_str_arg

from regularizer import LatentRegularizer
from target_flow_lib import DecoupledOTCFM, OTFlow 


FLAGS = flags.FLAGS

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


## statistics used to standardize inputs
channel_stats = {
    'cifar10': dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
    'cifar100': dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
    'mini_imgnet': dict(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                        std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
}

sizes = {
    'cifar10': 32,
    'cifar100': 32,
    'mini_imgnet': 84
}

padding = {
    'cifar10': 4,
    'cifar100': 4,
    'mini_imgnet': 8
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class AsymmetricDualViewTransform:
    """
    Verified DINO / BYOL augmentation strategy for CIFAR-10 (32x32).
    Generates View 1 (Almost always Blurred) and View 2 (Sometimes Solarized).
    """
    def __init__(self, size=32, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
        
        # SOTA Color Jitter parameters (Brightness, Contrast, Saturation, Hue)
        color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        
        # CIFAR-10 specific Blur (kernel size must be odd and ~10% of image size)
        blur_kernel = 3
        # Sigma range from BYOL/DINO specs
        blur = T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))

        # View 1: Always blurred, no solarization
        self.transform_1 = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([blur], p=0.999), ### p=0.9 before
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        # View 2: Rarely blurred
        self.transform_2 = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([blur], p=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        """Yields [View 1, View 2] for the LeJEA prediction loss."""
        return [self.transform_1(x), self.transform_2(x)]



def get_transforms(dataset, aug):

    size = sizes[dataset]
    pad = padding[dataset]
    stats = channel_stats[dataset]
    if aug == 'none':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((size, size)),
                                        transforms.Normalize(**stats)])
    elif aug == 'simclr':
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(**stats)])
    elif aug == 'laplace_strong':
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=pad, padding_mode="reflect"),
            RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize(**stats)
        ])
    elif aug == 'laplace_weak':
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=pad, padding_mode="reflect"),
            RandAugment(1),
            transforms.ToTensor(),
            transforms.Normalize(**stats)
        ])
    elif aug == 'lejea':
        # This returns the callable class that yields a list of 2 asymmetric views
        transform = AsymmetricDualViewTransform(size=size, **stats)
    else:
        print('No Augmentation Found')
        exit()

    return transform


###! infinite iterator: iteration based train 
def cycle(iterable):
    while True:
        for aug_images, target in iterable:
            yield aug_images, target


##* Add is_distributed and DistributedSampler to get_dataloaders
def get_dataloaders(dataset, bsz, aug='none', is_distributed=False):
    # Route the Transform Logic
    transform = get_transforms(dataset, aug)
    
    
    if dataset == 'cifar10':
        # Automatically downloads CIFAR-10 to ./data/cifar10
        train_data = torchvision.datasets.CIFAR10(
            root='./data', train=True, transform=transform, download=True
        )
        test_data = torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=transform, download=True
        )
    elif dataset == 'cifar100':
        train_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar100/by-image/train+val',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar100/by-image/test',
            transform = transform
        )
    elif dataset == 'mini_imgnet':
        train_data = torchvision.datasets.ImageFolder(
            root = '/content/data/images/miniimagenet/train',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = '/content/data/images/miniimagenet/test',
            transform = transform
        )
    else:
        print('Dataset Not Supported')
        exit()

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = bsz,
        shuffle=True,
        num_workers = 1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = bsz,
        shuffle=True,
        num_workers = 1
    )

    # Add DistributedSampler to partition data across GPUs
    train_sampler = DistributedSampler(train_data) if is_distributed else None
    test_sampler = DistributedSampler(test_data, shuffle=False) if is_distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = bsz,
        shuffle=(train_sampler is None), # Disable standard shuffle if using sampler
        sampler=train_sampler,
        num_workers = 2, # Increased for better IO
        pin_memory=True  # Important for GPU transfer speeds
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = bsz,
        shuffle=False,
        sampler=test_sampler,
        num_workers = 2,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler


def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config, workdir, local_rank=0, global_rank=0, is_distributed=False):
    """Runs the pure Flow Matching training pipeline for Representation Learning."""
    
    if global_rank == 0:
        logging.info('Starting Training Pipeline')
    
    ## different seed per GPU
    set_seed(config.training.seed + global_rank)
    
    ## define loss function type
    loss_type = getattr(config.training, 'loss_type', 'csfm').lower()
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        # Mac test mode: DDP doesn't support MPS well yet, fallback to CPU if distributed
        device = torch.device("cpu") if is_distributed else torch.device("mps")
    else:
        device = torch.device("cpu")
    
    config.device = device
    
    # 2. Dataset Setup
    args = cli.parse_commandline_args()
    args.dataset = config.data.dataset
    
    train_loader, eval_loader, train_sampler = get_dataloaders(
        args.dataset, args.batch_size, config.data.aug, is_distributed
    )
    train_iter = iter(cycle(train_loader))
    eval_iter = iter(cycle(eval_loader))

    # 3. Model & DDP Setup
    score_model = mutils.create_model(config).float().to(device)
    
    #* Extract the raw module up front for safe tracking
    model_module = score_model
    
    ##* log the network's parameter size
    if global_rank == 0:
        # 1. Count Total Parameters
        total_params = sum(p.numel() for p in model_module.parameters())
        
        # 2. Safely Count Encoder Parameters
        if hasattr(model_module, 'encoder') and model_module.encoder is not None:
            # We must call .parameters() on the module to iterate over its weights
            encoder_params = sum(p.numel() for p in model_module.encoder.parameters() if p.requires_grad)
        else:
            encoder_params = 0
            
        # 3. Calculate the rest (U-Net)
        decoder_params = total_params - encoder_params 
        
        logging.info("=" * 40)
        logging.info("Neural Network Architecture Setup:")
        logging.info(f"Total Parameters:    {total_params / 1e6:.2f} M")
        if encoder_params > 0:
            logging.info(f"Encoder (WideResNet): {encoder_params / 1e6:.2f} M")
            logging.info(f"Decoder (U-Net):     {decoder_params / 1e6:.2f} M")
        logging.info("=" * 40)
    
    ##! LeJEA parameter freezing
    if loss_type == 'lejea':
        # 1. Freeze the entire U-Net / Model completely
        for param in model_module.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze ONLY the WideResNet encoder
        for param in model_module.encoder.parameters():
            param.requires_grad = True
            
        if 'global_rank' not in locals() or global_rank == 0:
            logging.info("LeJEPA Mode Active: U-Net is entirely frozen. Only WideResNet will be trained.")

    
    ##! NEW CHANGE: LR ENCODER 
    # Filter out frozen parameters. 
    # By casting to a list, we can pass it to both EMA and the Optimizer safely.
    trainable_params = list(filter(lambda p: p.requires_grad, model_module.parameters()))
    
    # EMA and Optimizer must attach to the underlying raw model parameters BEFORE DDP and Compile.
    # This guarantees they track the true mathematical weights, immune to wrapper interference.
    ema = ExponentialMovingAverage(trainable_params, decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model_module)


    if is_distributed:
        if torch.cuda.is_available():
            score_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(score_model)
            # Wrap the model in DDP
            score_model = DDP(score_model, device_ids=[local_rank], output_device=local_rank)
        else:
            score_model = DDP(score_model) # Mac/CPU testing mode
            
        # Re-assign model_module to point inside the DDP wrapper for saving checkpoints later
        model_module = score_model.module


    if torch.cuda.is_available():
        if global_rank == 0:
            logging.info("Optimizing model with torch.compile (TorchInductor)...")
        
        # Check if GPU is newer (Compute Capability 8.0+ like A100)
        if torch.cuda.get_device_capability()[0] >= 8:
            score_model = torch.compile(score_model, mode="reduce-overhead")
        else:
            # Fallback for T4/V100 GPUs on standard Colab
            score_model = torch.compile(score_model, mode="default")


    ##* initialize state securely
    # We explicitly include step=0 so it never throws a KeyError
    state = dict(optimizer=optimizer, model=model_module, ema=ema, step=0)
    initial_step = 1


    ##* auto-resume logic: detect and load the latest checkpoint
    checkpoint_dir = os.path.join(workdir, "checkpoints_" + config.training.experiment_name)
    checkpoint_enc_dir = os.path.join(workdir, "checkpointsenc_" + config.training.experiment_name) 
    
    ckpt_pattern = os.path.join(checkpoint_dir, 'checkpoint_*.pth')
    ckpt_files = glob.glob(ckpt_pattern)
    
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        if global_rank == 0:
            logging.info(f"Resuming training from: {latest_ckpt}")
            
        state = restore_checkpoint(latest_ckpt, state, config.device)
        
        # Safely extract the step, defaulting to 0 if missing or None
        saved_step = state.get('step', 0)
        if saved_step is None:
            saved_step = 0
            
        initial_step = int(saved_step) + 1
        
        if getattr(config.training, 'include_encoder', False):
            save_step = int(saved_step) // config.training.snapshot_freq
            enc_ckpt_path = os.path.join(checkpoint_enc_dir, f'encoder_state_{save_step}.pth')
            if os.path.exists(enc_ckpt_path):
                if global_rank == 0:
                    logging.info(f"Restoring encoder weights from: {enc_ckpt_path}")
                loaded_enc = torch.load(enc_ckpt_path, map_location=config.device, weights_only=False)
                model_module.encoder.load_state_dict(loaded_enc)
    else:
        if global_rank == 0:
            logging.info("No existing checkpoints found. Starting training from scratch.")



    ##* Initiate Flow Engine
    # AE and LeJEA do not use structural degradation or vector fields.
    if loss_type in ['ae', 'lejea']:
        if global_rank == 0:
            logging.info(f"[model] {loss_type}. Bypassing Flow Engine initialization.")
        flow_engine = None
        
    # CSFM, OT, and CDAE rely on degradation physics
    elif loss_type in ['csfm', 'ot', 'cdae']:
        
        # Determine the exact physics engine to use. 
        # (CDAE can simulate either OT or CSFM physics based on your config)
        if loss_type == 'cdae':
            physics_mode = getattr(config.training, 'cdae_degradation', 'ot').lower()
            logging.info(f"[model] CDAE - {physics_mode}")
        else:
            physics_mode = loss_type # 'csfm' or 'ot'
        
        
        if physics_mode == 'csfm':
            decay_mode = getattr(config.training, 'decay_mode', 'exponential').lower()
            
            k_base = getattr(config.training, 'k_base', 6.0)
            k_beta = getattr(config.training, 'k_beta', 0.7)
            t1_base = getattr(config.training, 't1', 0.8)
            
            t0_base = getattr(config.training, 't0_base', 0.35)
            t0_beta = getattr(config.training, 't0_beta', 0.7)
            s_steep = getattr(config.training, 's_steep', 10.0)
            
            flow_engine = DecoupledOTCFM(
                img_shape=(config.data.num_channels, config.data.image_size, config.data.image_size),
                device=config.device,
                operators=config.training.operators,
                decay_mode=decay_mode,
                # Exponential arguments
                k_base=k_base, k_beta=k_beta,
                # Truncated arguments
                t0_base=t0_base, t0_beta=t0_beta, s_steep=s_steep,
                t1_base= t1_base
            )
            
            if global_rank == 0:
                logging.info(f"[model] CSFM Decoupled Physics Engine Loaded.")
                logging.info(f"        -> Decay Mode: {decay_mode.upper()}")
                logging.info(f"        -> Active Operators: {config.training.operators}")
                
                if decay_mode in ['exponential', 'piecewise_linear']:
                    logging.info(f"        -> k_base: {k_base} | k_beta: {k_beta}")
                elif decay_mode == 'truncated':
                    logging.info(f"        -> t0_base: {t0_base} | t0_beta: {t0_beta} | s_steep: {s_steep}")
                elif decay_mode == 'hard_truncated_exponential':
                    logging.info(f"        -> k_base: {k_base} | k_beta: {k_beta} | t1_base {t1_base}")
        
        elif physics_mode == 'ot':
            flow_engine = OTFlow(device=config.device)
            if global_rank == 0:
                logging.info("[model Standard OT Physics Engine.")
        else:
            raise ValueError(f"Unknown physics mode: {physics_mode}")
    else:
        raise ValueError(f"Unknown loss_type in config: {loss_type}")


    # 4. Initialize Regularizer (Maximum Entropy Constraint)
    latent_dim = int(64 * config.model.widen_factor)
    regularizer = LatentRegularizer(
        feature_dim=latent_dim, 
        mode=getattr(config.training, 'reg_mode', 'sigreg'),
        reduce_mean=config.training.reduce_mean
    ).to(config.device)

    # 5. Checkpoint Directories: only Rank 0
    if global_rank == 0:
        checkpoint_dir = os.path.join(workdir, f"checkpoints_{config.training.experiment_name}")
        checkpoint_enc_dir = os.path.join(workdir, f"checkpointsenc_{config.training.experiment_name}")
        tf.io.gfile.makedirs(checkpoint_dir)
        tf.io.gfile.makedirs(checkpoint_enc_dir)

    # 6. Build the Step Functions
    optimize_fn = losses.optimization_manager(config)
    
    train_step_fn = losses.get_step_fn(
        flow_engine=flow_engine, regularizer=regularizer, train=True,
        optimize_fn=optimize_fn, reduce_mean=config.training.reduce_mean, config=config
    )
    eval_step_fn = losses.get_step_fn(
        flow_engine=flow_engine, regularizer=regularizer, train=False,
        optimize_fn=optimize_fn, reduce_mean=config.training.reduce_mean, config=config
    )

    # 7. Core Training Loop
    num_train_steps = config.training.n_iters
    
    if global_rank == 0:
        
        time_mode = getattr(config.training, 'time_sampling_mode', 'uniform')
        tc = getattr(config.training, 't0', 1e-8)
        weight_mode = getattr(config.training, 'loss_weighting_mode', 'none')
        
        logging.info(f"[model] regularization: {getattr(config.training, 'reg_mode', 'sigreg')}, lambda: {getattr(config.training, 'lambda_reg', 0.1)}")
        logging.info(f"[model] time_sampling_mode - {time_mode} | t0_encoder - {tc} | weight mode - {weight_mode}")
        logging.info(f"Starting training loop at step {initial_step}.")
        
    for step in range(initial_step, num_train_steps + 1):
        
        # Tell the sampler the current epoch so it shuffles mathematically
        if is_distributed and step % len(train_loader) == 0:
            train_sampler.set_epoch(step // len(train_loader))
        
        # Fetch data
        images, labels = next(train_iter)
        
        # Safely push either the single batch or the dual-view list to the GPU
        if isinstance(images, list):
            batch = [img.to(config.device, non_blocking=True) for img in images]
        else:
            batch = images.to(config.device, non_blocking=True)
            
        
        ##* Optimize with bfloat16 Mixed Precision
        # This context manager automatically handles the safe downcasting of operations
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict = train_step_fn(state, batch)
        
        # Synchronize Loss for correct logging
        if is_distributed:
            dist.all_reduce(loss_dict['total'], op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_dict['task'], op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_dict['raw_reg'], op=dist.ReduceOp.AVG)
        
        # Synchronize Loss for correct logging
        if is_distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        
        
        # Asymmetric Logging and Saving (Rank 0 Only)
        if global_rank == 0:
            
            if step % config.training.log_freq == 0:
                logging.info(
                    "[train] "
                    f"step: {step}, total_loss: {loss_dict['total'].item():.4e} | "
                    f"task_loss: {loss_dict['task'].item():.4e} | "
                    f"raw_reg: {loss_dict['raw_reg'].item():.4e}"
                )
                
            if step % config.training.eval_freq == 0:
                
                eval_images, _ = next(eval_iter)
                if isinstance(eval_images, list):
                    eval_batch = [img.to(config.device, non_blocking=True) for img in eval_images]
                else:
                    eval_batch = eval_images.to(config.device, non_blocking=True)
                
                # Evaluate with bfloat16 to match training precision and speed up evaluation
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    eval_dict = eval_step_fn(state, eval_batch)
                
                if is_distributed:
                    dist.all_reduce(eval_dict['total'], op=dist.ReduceOp.AVG)
                    dist.all_reduce(eval_dict['task'], op=dist.ReduceOp.AVG)
                    dist.all_reduce(eval_dict['raw_reg'], op=dist.ReduceOp.AVG)
                
                logging.info(
                    "[eval] "
                    f"step: {step}, total_loss: {eval_dict['total'].item():.4e} | "
                    f"task_loss: {eval_dict['task'].item():.4e} | "
                    f"raw_reg: {eval_dict['raw_reg'].item():.4e}"
                )
            if step % config.training.snapshot_freq == 0 or step == num_train_steps:
                save_step = step // config.training.snapshot_freq
                # Save the unwrapped module so it can be loaded on a single GPU later
                state_to_save = dict(optimizer=optimizer, model=model_module, ema=ema, step=step)
                save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state_to_save)

                if getattr(config.training, 'include_encoder', False):
                    encoder_state = model_module.encoder.state_dict()
                    torch.save(encoder_state, os.path.join(checkpoint_enc_dir, f'encoder_state_{save_step}.pth'))