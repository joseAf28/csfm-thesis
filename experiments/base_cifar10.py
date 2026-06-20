import ml_collections
import torch

def get_base_config():
    config = ml_collections.ConfigDict()
    
    
    # Shared Training Settings
    config.training = training = ml_collections.ConfigDict()
    training.seed = 0
    training.batch_size = 64        
    training.n_iters = 40000            # The baseline target
    
    # Save a checkpoint to Google Drive 
    training.snapshot_freq = 20000    
    
    # Do NOT log every step. It will crash the Colab UI.
    training.log_freq = 300           
    
    # Evaluation (generating samples/calculating metrics) 
    training.eval_freq = 1400
    
    # Evaluation representation and sampling
    training.repr_eval_freq = 40000
    
    
    training.snapshot_freq_for_preemption = 10000
    training.include_encoder = True
    training.probabilistic_encoder = False ##! Old code: Not considered 
    training.reduce_mean = False
    
    ##* -- Legacy definitions
    training.snapshot_sampling = False 
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.experiment_name = ''
    training.lambda_reconstr = 0.0
    training.apply_mixup = False
    training.recon = 'l2'

    training.lambda_z = 0.0
    ##* --
    
    # --- Shared Evaluation Settings ---
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 0
    evaluate.end_ckpt = 5
    evaluate.batch_size = 1024
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    
    # --- Shared Data Settings ---
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'cifar10'
    data.image_size = 32
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    
    model.name = 'ncsnpp'
    
    model.widen_factor = 4. ##* more depth to the encoder (prev 2.0)
    
    
    model.fourier_scale = 16
    model.scale_by_sigma = False ##! False for Flow Matching 
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'    
    
    #* The U-Net Capacity
    # Upgraded from 16 to 64. 
    # This squares the number of parameters in the vector field predictor, 
    # allowing it to track the complex, overlapping exponential decays 
    # of your spatial, color, and frequency operators. (prev 16)
    model.nf = 64
    
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False  
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.0
    model.conv_size = 3
    
    
    ##* -- Legacy definitions SDE Parameters (Keep for Control)
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.constrained_architecture = False
    ##* -- 

    # Shared Optimization Settings
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config