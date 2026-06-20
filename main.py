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
##* Code adapted from the Google Repository 

"""Training and evaluation"""

import glob
import torch
import os
import sys
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import tensorflow as tf
import run_lib
import torch.distributed as dist
import gc

import eval_metrics
import sample


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")

flags.DEFINE_enum("mode", None, ["train", "eval", "sample", "all"], "Running mode: train, eval, sample, or all")

flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.DEFINE_string("checkpoint", None, "Path to the frozen checkpoint file for evaluation/sampling.")

flags.DEFINE_string("h5_path", "embeddings/features.h5", "Path to store/load HDF5 features")
flags.DEFINE_string("out_dir", "samples", "Directory to save the generated images")


flags.DEFINE_string("sample_method", "heun", "Numerical ODE solver method (euler/heun)")
flags.DEFINE_integer("sample_steps", 50, "Integration step")
flags.DEFINE_float("sample_rho", 1.0, "The rho curvature value for the ODE solver")

flags.DEFINE_integer("sample_batch_size", 64, "Number of images to generate per batch")
flags.DEFINE_integer("sample_fid_samples", 1000, "Number of images to generate for Geodesic FID calculation")
flags.DEFINE_enum("sample_condition_mode", "conditional", ["unconditional", "conditional", "all"], "Mode for latent z sampling (noise prior vs real encoded features).")

flags.mark_flags_as_required(["workdir", "config", "mode"])



def main(argv):

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        if torch.cuda.is_available():
            backend = "nccl" # Standard for NVIDIA Clusters
            torch.cuda.set_device(local_rank)
        else:
            backend = "gloo" # Fallback for Mac CPU distributed testing
        
        dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)
    
    # Create the working directory ONLY on Rank 0 to avoid race conditions
    if global_rank == 0:
        FLAGS.workdir = f'{FLAGS.workdir}'
        tf.io.gfile.makedirs(FLAGS.workdir)
            
        log_filename = f'stdout_{FLAGS.mode}.txt'
        gfile_stream = open(os.path.join(FLAGS.workdir, log_filename), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
            
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(handler)
            
        logger.setLevel('INFO')
        
    if is_distributed:
        dist.barrier()
    

    if FLAGS.mode == "all":
        
        SAMPLING_FLAG = False
        
        # Capture the absolute end goal from the config
        total_train_iters = FLAGS.config.training.n_iters
        eval_interval = FLAGS.config.training.repr_eval_freq
        
        # Loop: 30k, 60k, 90k...
        for target_iters in range(eval_interval, total_train_iters + eval_interval, eval_interval):
            # Ensure we cap the cycle at the actual maximum iterations
            target_iters = min(target_iters, total_train_iters)
            
            if global_rank == 0:
                logging.info(f"\n{'='*60}\nStarting Orchestration Cycle: Up to {target_iters} steps\n{'='*60}")
                
            # 1. Unlock the config and set the temporary horizon for run_lib
            with FLAGS.config.unlocked():
                FLAGS.config.training.n_iters = target_iters
                if global_rank == 0:
                    FLAGS.config.training.seed += int(os.environ.get('SLURM_PROCID', 0))

            ##* train
            logging.info(f"Starting Phase 1: Training to {target_iters}")
            run_lib.train(FLAGS.config, FLAGS.workdir, local_rank, global_rank, is_distributed)
            
            # Force cleanup of memory to prevent OOM across cycles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if is_distributed:
                dist.barrier()
                
            # auto-discover checkoint
            ckpt_dir = os.path.join(FLAGS.workdir, "checkpoints_" + FLAGS.config.training.experiment_name)
            ckpt_files = glob.glob(os.path.join(ckpt_dir, 'checkpoint_*.pth'))
            
            if not ckpt_files:
                raise FileNotFoundError(f"Could not find any checkpoints in {ckpt_dir}! Did training fail?")
                
            resolved_ckpt_path = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            if global_rank == 0:
                logging.info(f"Cycle Checkpoint Discovered: {resolved_ckpt_path}")
                
            # dynamic path routing 
            # Append the current step so you don't overwrite the 30k files with the 60k files
            cycle_h5_path = os.path.join(FLAGS.workdir, f"features_step{target_iters}.h5")
            cycle_out_dir = os.path.join(FLAGS.workdir, f"samples_step{target_iters}")
            
            if global_rank == 0:
                os.makedirs(cycle_out_dir, exist_ok=True)
                
            ##* evaluation
            if global_rank == 0:
                logging.info(f"Starting Phase 2: Evaluation Pipeline. Saving to {cycle_h5_path}")
                eval_metrics.run_eval_pipeline(
                    config=FLAGS.config, 
                    ckpt_path=resolved_ckpt_path,
                    h5_path=cycle_h5_path
                )
            
            #! No sampling  during traing
            ##* sampling 
            if global_rank == 0:
                
                loss_type = getattr(FLAGS.config.training, 'loss_type', 'csfm').lower()
                
                if loss_type in ['csfm', 'ot'] and SAMPLING_FLAG:
                    
                    logging.info(f"Starting Phase Sampling. Saving to {auto_out_dir}")
                    
                    _, eval_loader, _ = run_lib.get_dataloaders(
                        dataset=FLAGS.config.data.dataset, 
                        bsz=FLAGS.sample_batch_size, 
                        aug='none', 
                        is_distributed=False
                    )
                    
                    sample.run_sampling_tests(
                        config=FLAGS.config,
                        workdir=FLAGS.workdir,
                        ckpt_path=resolved_ckpt_path, 
                        out_dir=cycle_out_dir,
                        method=FLAGS.sample_method,
                        steps=FLAGS.sample_steps,  
                        rho=FLAGS.sample_rho, 
                        eval_loader=eval_loader,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        condition_mode=FLAGS.sample_condition_mode
                    )
                else:
                    logging.error(f"Sampling Inactive")
        
            
            # Re-sync distributed workers before starting the next cycle
            if is_distributed:
                dist.barrier()
                
    ##* isolated names
    else:
        # Keep original single-execution behavior for standard debugging
        if FLAGS.mode == "train":
            if global_rank == 0:
                FLAGS.config.training.seed += int(os.environ.get('SLURM_PROCID', 0))
            logging.info("Starting Phase 1: Training")
            run_lib.train(FLAGS.config, FLAGS.workdir, local_rank, global_rank, is_distributed)

        ## auto-discover latest checkpoint for isolated eval/sample
        resolved_ckpt_path = FLAGS.checkpoint
        if not resolved_ckpt_path and FLAGS.mode in ["eval", "sample"]:
            ckpt_dir = os.path.join(FLAGS.workdir, "checkpoints_" + FLAGS.config.training.experiment_name)
            ckpt_files = glob.glob(os.path.join(ckpt_dir, 'checkpoint_*.pth'))
            if not ckpt_files:
                raise FileNotFoundError(f"Could not find any checkpoints in {ckpt_dir}!")
            resolved_ckpt_path = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            if global_rank == 0:
                logging.info(f"Auto-discovered latest checkpoint: {resolved_ckpt_path}")

        auto_h5_path = os.path.join(FLAGS.workdir, "features.h5")
        auto_out_dir = os.path.join(FLAGS.workdir, "samples")
        
        if global_rank == 0:
            os.makedirs(auto_out_dir, exist_ok=True)
            
        if FLAGS.mode == "eval":
            if global_rank == 0:
                logging.info(f"Starting Phase 2: Evaluation Pipeline. Saving to {auto_h5_path}")
                eval_metrics.run_eval_pipeline(
                    config=FLAGS.config, 
                    ckpt_path=resolved_ckpt_path,
                    h5_path=auto_h5_path
                )
        
        if FLAGS.mode == "sample":
            if global_rank == 0:
                loss_type = getattr(FLAGS.config.training, 'loss_type', 'csfm').lower()
                
                if loss_type in ['csfm', 'ot']:
                    
                    logging.info(f"Starting Sampling. Saving to {auto_out_dir}")
                    
                    _, eval_loader, _ = run_lib.get_dataloaders(
                        dataset=FLAGS.config.data.dataset, 
                        bsz=FLAGS.sample_batch_size, 
                        aug='none', 
                        is_distributed=False
                    )
                    
                    sample.run_sampling_tests(
                        config=FLAGS.config,
                        workdir=FLAGS.workdir,
                        ckpt_path=resolved_ckpt_path, 
                        out_dir=auto_out_dir,
                        method=FLAGS.sample_method,
                        steps=FLAGS.sample_steps,  
                        rho=FLAGS.sample_rho, 
                        eval_loader=eval_loader,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        condition_mode=FLAGS.sample_condition_mode
                    )
                else:
                    logging.error(f"Cannot run sampling on loss_type '{loss_type}'.")
        
        
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    app.run(main)
