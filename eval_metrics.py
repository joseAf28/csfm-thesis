import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from models import utils as mutils
from utils import restore_checkpoint
from run_lib import get_dataloaders


##* feature extraction
def extract_and_save_features(model, dataloader, device, h5_file, split_name):
    """Passes the dataset strictly through the frozen encoder and saves z to HDF5."""
    # use eval mode (locks BatchNorm/Dropout)
    model.encoder.eval() 
    
    all_z = []
    all_y = []
    
    logging.info(f"Extracting features for {split_name} split...")
    
    with torch.inference_mode(): 
        for images, labels in tqdm(dataloader, desc=f"Extracting {split_name}"):
            images = images.to(device)
            
            # Directly query the Wide-ResNet
            encoder_output = model.encoder(images, t=None)
            # Extract representation z
            z = encoder_output[2]  
            
            all_z.append(z.cpu().numpy())
            all_y.append(labels.cpu().numpy())
            
    # Concatenate all batches
    z_array = np.concatenate(all_z, axis=0)
    y_array = np.concatenate(all_y, axis=0)
    
    # Save to HDF5
    h5_file.create_dataset(f"{split_name}_z", data=z_array, compression="gzip")
    h5_file.create_dataset(f"{split_name}_y", data=y_array, compression="gzip")
    logging.info(f"Saved {z_array.shape[0]} samples to {split_name} split.")


##* evaluation metrics
def train_linear_probe(train_z, train_y, test_z, test_y, num_classes=10, epochs=100, lr=1e-3, wd=1e-4):
    """Trains a linear classifier on frozen representations."""
    device = train_z.device
    d = train_z.shape[1]
    
    linear_classifier = nn.Linear(d, num_classes).to(device)
    optimizer = torch.optim.AdamW(linear_classifier.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(train_z, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    linear_classifier.train()
    for epoch in tqdm(range(epochs), desc="Linear Probe Epochs", leave=False):
        for z_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = linear_classifier(z_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
    linear_classifier.eval()
    with torch.no_grad():
        logits = linear_classifier(test_z)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == test_y).float().mean().item() * 100
        
    return acc


def evaluate_knn(train_z, train_y, test_z, test_y, k=20):
    """k-Nearest Neighbors using Cosine Similarity."""
    train_z_norm = F.normalize(train_z, p=2, dim=1).numpy()
    test_z_norm = F.normalize(test_z, p=2, dim=1).numpy()
    
    logging.info(f"Fitting k-NN (k={k})...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_z_norm, train_y.numpy())
    acc = knn.score(test_z_norm, test_y.numpy()) * 100
    return acc


def evaluate_silhouette(z, y, num_samples=5000):
    """Mathematical measure of geometric cluster tightness."""
    # Running Silhouette on full dataset is O(N^2). Use stratified subset.
    num_samples = min(num_samples, z.shape[0])
    indices = torch.randperm(z.shape[0])[:num_samples]
    
    z_sub = z[indices].numpy()
    y_sub = y[indices].numpy()
    
    logging.info(f"Computing Silhouette Score on {num_samples} samples...")
    score = silhouette_score(z_sub, y_sub, metric='cosine')
    return score



##* eval_metrics.py - Refactored Main Entry Point
def run_eval_pipeline(config, ckpt_path, h5_path, extract=True, eval_metrics_flag=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Ensure directory for h5 exists to prevent FileNotFoundError
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    ##* extract features 
    if extract:
        # 1. Turn OFF Data Augmentation for pure feature extraction
        train_loader, test_loader, _ = get_dataloaders(config.data.dataset, bsz=512, aug='none', is_distributed=False)
        
        # 2. Load Model & Checkpoint using the passed 'config'
        logging.info("Initializing unified model for extraction...")
        model = mutils.create_model(config).to(device)
        
        loaded_state = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        
        # Safely extract weights depending on how the checkpoint was saved
        if 'model' in loaded_state:
            # Load into the full model (which includes the encoder)
            model.load_state_dict(loaded_state['model'], strict=False)
        else:
            # Fallback if it's strictly an encoder state dictionary
            model.encoder.load_state_dict(loaded_state)
        
        # 3. Extract and Save to HDF5
        with h5py.File(h5_path, 'w') as h5f:
            extract_and_save_features(model, train_loader, device, h5f, "train")
            extract_and_save_features(model, test_loader, device, h5f, "test")
            
        logging.info(f"Feature extraction complete. Data saved to {h5_path}")

    ##* evaluation metrics
    if eval_metrics_flag:
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Cannot run evaluation. {h5_path} does not exist.")
            
        logging.info(f"Loading representations from {h5_path}...")
        with h5py.File(h5_path, 'r') as h5f:
            train_z = torch.tensor(h5f['train_z'][:])
            train_y = torch.tensor(h5f['train_y'][:], dtype=torch.long)
            test_z = torch.tensor(h5f['test_z'][:])
            test_y = torch.tensor(h5f['test_y'][:], dtype=torch.long)
            
        train_z_gpu, train_y_gpu = train_z.to(device), train_y.to(device)
        test_z_gpu, test_y_gpu = test_z.to(device), test_y.to(device)
        
        logging.info("--- Phase 2: Evaluation Suite Results ---")
        
        acc_100 = train_linear_probe(train_z_gpu, train_y_gpu, test_z_gpu, test_y_gpu)
        logging.info(f"Linear Probe (100% Data): {acc_100:.2f}%")
        
        indices_10 = torch.randperm(train_z.shape[0])[:5000]
        acc_10 = train_linear_probe(train_z_gpu[indices_10], train_y_gpu[indices_10], test_z_gpu, test_y_gpu)
        logging.info(f"Few-Shot Probe (10% Data): {acc_10:.2f}%")
        
        indices_1 = torch.randperm(train_z.shape[0])[:500]
        acc_1 = train_linear_probe(train_z_gpu[indices_1], train_y_gpu[indices_1], test_z_gpu, test_y_gpu)
        logging.info(f"Few-Shot Probe  (1% Data): {acc_1:.2f}%")
        
        acc_knn = evaluate_knn(train_z, train_y, test_z, test_y, k=20)
        logging.info(f"k-NN (k=20) Accuracy:     {acc_knn:.2f}%")
        
        sil_score = evaluate_silhouette(test_z, test_y)
        logging.info(f"Silhouette Score (-1 to 1): {sil_score:.4f}")