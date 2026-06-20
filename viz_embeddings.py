import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap


def plot_correlation_matrix(h5_path, name_file, split="test"):
    print(f"Loading {split} features from {h5_path} for Correlation analysis...")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}. Run extraction first.")
        
    with h5py.File(h5_path, 'r') as f:
        z = f[f'{split}_z'][:]

    print("Computing empirical correlation matrix...")
    # Calculate the correlation matrix (shape: latent_dim x latent_dim)
    # rowvar=False means columns are the variables (dimensions)
    corr_matrix = np.corrcoef(z, rowvar=False)
    
    # Extract the off-diagonal elements to calculate the metric
    latent_dim = corr_matrix.shape[0]
    mask = ~np.eye(latent_dim, dtype=bool) # Boolean mask for off-diagonals
    off_diagonals = corr_matrix[mask]
    
    mean_abs_corr = np.abs(off_diagonals).mean()
    max_abs_corr = np.abs(off_diagonals).max()
    
    print(f"--- Decorrelation Metrics ---")
    print(f"Mean Absolute Off-Diagonal Correlation: {mean_abs_corr:.4f}")
    print(f"Max Absolute Off-Diagonal Correlation:  {max_abs_corr:.4f}")

    # Plotting the Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a diverging colormap: Red for +1, Blue for -1, White for 0
    cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_title(f'Latent Correlation Matrix\nMean Abs Off-Diagonal: {mean_abs_corr:.4f}', fontsize=14)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Latent Dimension')
    
    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation Coefficient')
    
    plt.tight_layout()
    
    out_name = os.path.join(os.path.dirname(h5_path), f'correlation_matrix_{split}_{name_file}.png')
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Saved Correlation Matrix plot to {out_name}")
    plt.close()
    
    

def plot_svd_spectrum(h5_path, name_file, split="test"):
    print(f"Loading {split} features from {h5_path} for SVD analysis...")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}. Run extraction first.")
        
    with h5py.File(h5_path, 'r') as f:
        z = f[f'{split}_z'][:]

    print("Computing SVD / PCA...")
    # Using PCA to get the singular values and explained variance
    pca = PCA()
    pca.fit(z)
    
    explained_variance = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    
    # Calculate "Effective Dimensionality" (How many dims explain 90% of variance)
    cumulative_variance = np.cumsum(explained_variance)
    eff_dim_90 = np.argmax(cumulative_variance >= 0.90) + 1
    total_dims = z.shape[1]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Explained Variance Ratio (Log Scale)
    ax1.plot(range(1, total_dims + 1), explained_variance, color='b', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title('SVD Spectrum (Explained Variance)', fontsize=14)
    ax1.set_xlabel('Principal Component Rank')
    ax1.set_ylabel('Explained Variance Ratio (Log Scale)')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 2: Cumulative Variance
    ax2.plot(range(1, total_dims + 1), cumulative_variance, color='r', linewidth=2)
    ax2.axhline(y=0.90, color='k', linestyle=':', label='90% Variance')
    ax2.axvline(x=eff_dim_90, color='k', linestyle=':', label=f'90% at rank {eff_dim_90}')
    ax2.set_title(f'Cumulative Variance (Effective Dims: {eff_dim_90}/{total_dims})', fontsize=14)
    ax2.set_xlabel('Principal Component Rank')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.5)
    
    plt.suptitle(f'Dimensional Collapse Analysis ({split.capitalize()} Set)', fontsize=18, y=1.05)
    plt.tight_layout()
    
    out_name = os.path.join(os.path.dirname(h5_path), f'svd_spectrum_{split}_{name_file}.png')
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Saved SVD plot to {out_name}")
    plt.close()


def plot_side_by_side_manifold(h5_path, name_file, split="test", num_samples=5000):
    print(f"Loading {split} features from {h5_path}...")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}. Run extraction first.")
        
    with h5py.File(h5_path, 'r') as f:
        z = f[f'{split}_z'][:]
        y = f[f'{split}_y'][:]

    # Subsample 
    num_samples = min(num_samples, z.shape[0])
    indices = np.random.choice(z.shape[0], num_samples, replace=False)
    z_sub = z[indices]
    y_sub = y[indices]

    print(f"Computing t-SNE on {num_samples} samples...")
    tsne_reducer = TSNE(n_components=2, metric='cosine', init='pca', random_state=42)
    z_tsne = tsne_reducer.fit_transform(z_sub)

    print(f"Computing UMAP on {num_samples} samples...")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    z_umap = umap_reducer.fit_transform(z_sub)

    # Plotting Side-by-Side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot t-SNE
    scatter1 = ax1.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y_sub, cmap='tab10', s=10, alpha=0.7, edgecolors='none')
    ax1.set_title(f't-SNE Projection ({split.capitalize()} Set)', fontsize=16)
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot UMAP
    scatter2 = ax2.scatter(z_umap[:, 0], z_umap[:, 1], c=y_sub, cmap='tab10', s=10, alpha=0.7, edgecolors='none')
    ax2.set_title(f'UMAP Projection ({split.capitalize()} Set)', fontsize=16)
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Shared Colorbar
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right', fraction=0.02, pad=0.02)
    cbar.set_label('Classes', rotation=270, labelpad=15)
    
    plt.suptitle('Latent Manifold Representation Comparison', fontsize=20, y=1.02)
    
    out_name = os.path.join(os.path.dirname(h5_path), f'manifold_comparison_{split}_{name_file}.png')
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Path to features.h5")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    
    plot_side_by_side_manifold(args.h5_path, args.name, args.split, args.samples)
    plot_svd_spectrum(args.h5_path, args.name, args.split)
    plot_correlation_matrix(args.h5_path, args.name, args.split)