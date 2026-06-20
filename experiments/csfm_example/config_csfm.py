from expFinal.base_cifar10 import get_base_config

def get_config():
    config = get_base_config()
    
    config.training.experiment_name = 'CSFM'
    config.training.loss_type = 'csfm'
    
    config.training.ssd_eta = 0.0
    
    # Physics Settings
    config.training.operators = ['illumination', 'color', 'freq']
    config.training.r_low = 6
    
    config.training.k_base = 3.25
    config.training.k_beta = 0.9
    
    config.training.decay_mode = 'hard_truncated_exponential'
    config.training.t1 = 5e-1
    
    # Regularization
    config.training.reg_mode = 'sigreg'
    config.training.lambda_reg = 1e-3
    
    # Time sampling 
    config.training.time_sampling_mode = 'uniform'
    config.training.loss_weighting_mode = 'none'
    config.training.t0 = 5e-2
    
    ## encoder lr rate relative speed
    config.optim.encoder_lr_mult = 1.0

    # Augmentation
    config.data.aug = 'none'
    
    return config