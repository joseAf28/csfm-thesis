import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from training.target_flow import TargetCompositeFlow

def analyze_target_flow():
    # 1. Setup Configuration (CPU for local debugging)
    device = torch.device('cpu')
    batch_size = 32
    
    # 2. Get Real Data (CIFAR-10)
    print("Loading Data...")
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get one fixed batch of "Clean" images (x1)
    x1, _ = next(iter(loader))
    x1 = x1.to(device) # Shape: [32, 3, 32, 32]
    
    # 3. Initialize the Physics Engine
    # CIFAR-10 is 32x32, so we pass that shape.
    flow_engine = TargetCompositeFlow(img_shape=(3, 32, 32), device=device)
    
    # 4. Sweep through Time t from 0 (Noise) to 1 (Data)
    # Note: In this repo's convention, t=0 is Clean/Data and t=1 might be Noise? 
    # Let's verify standard Flow Matching: t=0 (Source/Noise) -> t=1 (Target/Data)
    # BUT check the code: run_lib.py passes t ~ [0,1].
    # target_flow.py logic: x_t starts at x0 (clean) and operators DEGRADE it.
    # So t=0 is CLEAN, t=1 is NOISY (Degraded). 
    # Flow direction is Clean -> Noise.
    
    time_steps = torch.linspace(0, 1, 50)
    
    magnitudes = []
    variances = []
    signal_energies = []
    
    print(f"\nAnalyzing Flow for {len(time_steps)} time steps...")
    print(f"{'Time':<10} | {'Vel Mag':<10} | {'Vel Var':<10} | {'Signal Energy'}")
    print("-" * 50)
    
    for t_val in time_steps:
        # Create time batch
        t = torch.ones(batch_size, 1, 1, 1).to(device) * t_val
        
        # Compute Target (Physics)
        # x_t: The noisy image at time t
        # u_t: The target vector field (velocity) at time t
        with torch.no_grad():
            x_t, u_t = flow_engine.compute_target(x1, t)
        
        # --- METRIC 1: Flow Magnitude ---
        # "How fast is the probability mass moving?"
        # L2 Norm per image, averaged over batch
        # u_t shape: [B, C, H, W] -> flatten to [B, D]
        u_flat = u_t.reshape(batch_size, -1)
        mag = torch.norm(u_flat, dim=1).mean().item()
        magnitudes.append(mag)
        
        # --- METRIC 2: Flow Variance ---
        # "How different are the flow vectors?"
        # Variance of the velocity components across all dimensions
        var = u_t.var().item()
        variances.append(var)
        
        # --- METRIC 3: Signal Energy (Norm of x_t) ---
        x_flat = x_t.reshape(batch_size, -1)
        energy = torch.norm(x_flat, dim=1).mean().item()
        signal_energies.append(energy)
        
        if t_val in [0.0, 0.5, 1.0]: # Print check points
            print(f"{t_val.item():.2f}       | {mag:.4f}     | {var:.4f}     | {energy:.4f}")


    # 5. Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(time_steps, magnitudes, label='Velocity Magnitude ||u_t||')
    plt.xlabel('Time t (0=Clean, 1=Noisy)')
    plt.title('Flow Magnitude')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(time_steps, variances, label='Velocity Variance', color='orange')
    plt.xlabel('Time t')
    plt.title('Flow Variance')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(time_steps, signal_energies, label='State Norm ||x_t||', color='green')
    plt.xlabel('Time t')
    plt.title('State Energy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('flow_analysis.png')
    print("\nAnalysis complete. Plot saved to 'flow_analysis.png'")

if __name__ == "__main__":
    analyze_target_flow()