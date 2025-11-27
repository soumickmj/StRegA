# StRegA Demo Workflow

This document provides comprehensive instructions for training and testing StRegA (Segmentation Regularised Anomaly) models for unsupervised anomaly detection in Brain MRIs using Compact Context-encoding Variational Autoencoder.

Based on the paper: ["StRegA: Unsupervised Anomaly Detection in Brain MRIs using a Compact Context-encoding Variational Autoencoder"](https://doi.org/10.1016/j.compbiomed.2022.106093)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing/Inference](#testinginference)
- [Paper Experiments](#paper-experiments)
- [Additional Features](#additional-features-in-repo)
- [Missing Features](#features-from-paper-not-in-repo)

---

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- Minimum 16GB RAM
- 50GB+ storage for datasets

### Software Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU support)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/soumickmj/StRegA.git
cd StRegA
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torchio nibabel h5py numpy scipy scikit-image matplotlib
pip install transformers  # For HuggingFace model
pip install wandb  # Optional: for experiment tracking
pip install tqdm pandas seaborn
```

### 3. Verify Installation

```python
import torch
import torchio
from ccevae import VAE
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Data Preparation

### Required Data Format

StRegA expects **FSL-segmented** brain MRI data. The preprocessing pipeline involves:

1. **Skull stripping** using FSL BET
2. **Brain tissue segmentation** using FSL FAST
3. **Resampling** to 256×256 slices

### Supported Datasets

The paper uses the following datasets:

| Dataset | Type | Description |
|---------|------|-------------|
| IXI | T1, T2, PD | Normal brain MRIs |
| MOOD | T1 | Medical Out-of-Distribution Detection challenge |
| BraTS | T1, T1ce, T2, FLAIR | Brain tumor segmentation (for testing) |

### Data Preprocessing Script

Create segmented data using FSL:

```bash
# Example FSL preprocessing
# 1. Skull stripping
bet input.nii.gz brain.nii.gz -f 0.5 -g 0

# 2. Tissue segmentation  
fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o segmented brain.nii.gz
```

### HDF5 Data Format

Training data should be stored in HDF5 format with the following structure:

```python
import h5py
import nibabel as nib
import numpy as np

# Example: Creating training HDF5 file
with h5py.File('training_data.h5', 'w') as f:
    for i, nifti_path in enumerate(nifti_files):
        data = nib.load(nifti_path).get_fdata()
        f.create_dataset(f'{i:05d}', data=data)
```

---

## Training

### Model Architecture

The cceVAE (Compact Context-encoding VAE) architecture:

- **Input size**: 256×256 (2D slices)
- **Latent dimension**: 1024
- **Feature map sizes**: (16, 64, 256, 1024)
- **Activation**: PReLU

### Training Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 100 | Number of training epochs |
| `lr` | 1e-4 | Learning rate |
| `z_dim` | 1024 | Latent space dimension |
| `ce_factor` | 0.5 | Context encoding loss weight |
| `beta` | 0.01 | KL divergence weight |
| `patch_size` | (256, 256, 1) | 2D slice size |
| `patches_per_volume` | 256 | Patches sampled per volume |

### Training Script

Create a training script `run_training.py`:

```python
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import torchio as tio

from ccevae import VAE
from helpers import kl_loss_fn, rec_loss_fn, geco_beta_update, get_square_mask
from dataloaders.ixi import IXITrainSet

# Configuration
config = {
    'gpu_id': "0",
    'seed': 1701,
    'batch_size': 16,
    'num_epochs': 100,
    'lr': 1e-4,
    'z_dim': 1024,
    'model_feature_map_sizes': (16, 64, 256, 1024),
    'ce_factor': 0.5,
    'beta': 0.01,
    'theta': 1.0,
    'use_geco': False,
    'patch_size': (256, 256, 1),
    'patches_per_volume': 256,
    'patchQ_len': 512,
    'save_path': './checkpoints',
    'train_id': 'ceVAE2D_brain'
}

# Set random seed
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
input_size = (1, 256, 256)
model = VAE(
    input_size=input_size,
    z_dim=config['z_dim'],
    fmap_sizes=config['model_feature_map_sizes'],
    conv_op=nn.Conv2d,
    tconv_op=nn.ConvTranspose2d,
    activation_op=torch.nn.PReLU
)
model.d = 2  # 2D model
model.to(device)

# Optimizer and scaler
optimizer = Adam(model.parameters(), lr=config['lr'])
scaler = GradScaler()

# Load datasets (modify paths as needed)
train_data_paths = {
    'ixi_t1': 'path/to/ixi_t1_segmented.hdf5',
    'ixi_t2': 'path/to/ixi_t2_segmented.hdf5',
    'ixi_pd': 'path/to/ixi_pd_segmented.hdf5',
    'mood_t1': 'path/to/mood_t1_seg.h5'
}

# Create dataset (example with IXI T1)
train_indices = list(range(100, 350))
trainset = IXITrainSet(
    indices=train_indices,
    data_path=train_data_paths['ixi_t1'],
    lazypatch=True
)

# Use TorchIO Queue for patch-based training
patch_queue = tio.data.Queue(
    subjects_dataset=trainset,
    max_length=config['patchQ_len'],
    samples_per_volume=config['patches_per_volume'],
    sampler=tio.data.UniformSampler(patch_size=config['patch_size']),
)

train_loader = DataLoader(
    dataset=patch_queue,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=0
)

# Training loop
vae_loss_ema = 1.0
theta = config['theta']

for epoch in range(config['num_epochs']):
    model.train()
    epoch_loss = 0.0
    
    for i, data in enumerate(train_loader):
        img = data['img']['data'].squeeze(-1)
        
        # Normalize
        tmp = img.view(img.shape[0], 1, -1)
        min_vals = tmp.min(2, keepdim=True).values
        max_vals = tmp.max(2, keepdim=True).values
        tmp = (tmp - min_vals) / max_vals
        x = tmp.view(img.size())
        
        # Remove NaN samples
        shape = x.shape
        tensor_reshaped = x.reshape(shape[0], -1)
        tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(), dim=1)]
        tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0], *shape[1:]).to(device)
        
        if tensor.shape[0] == 0:
            continue
            
        optimizer.zero_grad()
        
        # VAE forward pass
        with autocast():
            loss_vae = 0
            if config['ce_factor'] < 1:
                x_r, z_dist = model(tensor)
                kl_loss = kl_loss_fn(z_dist, sumdim=(1, 2, 3)) * config['beta']
                rec_loss_vae = rec_loss_fn(x_r, tensor, sumdim=(1, 2, 3))
                loss_vae = kl_loss + rec_loss_vae * theta
        
        # Context Encoding (CE) Part
        loss_ce = 0
        if config['ce_factor'] > 0:
            ce_tensor = get_square_mask(
                tensor.shape,
                square_size=(0, np.max(input_size[1:]) // 2),
                noise_val=(torch.min(tensor).item(), torch.max(tensor).item()),
                n_squares=(0, 3),
            )
            ce_tensor = torch.from_numpy(ce_tensor).float().to(device)
            inpt_noisy = torch.where(ce_tensor != 0, ce_tensor, tensor)
            
            with autocast():
                x_rec_ce, _ = model(inpt_noisy)
                rec_loss_ce = rec_loss_fn(x_rec_ce, tensor, sumdim=(1, 2, 3))
                loss_ce = rec_loss_ce
                loss = (1.0 - config['ce_factor']) * loss_vae + config['ce_factor'] * loss_ce
        else:
            loss = loss_vae
        
        # GECO update (optional)
        if config['use_geco'] and config['ce_factor'] < 1:
            g_goal = 0.1
            g_lr = 1e-4
            vae_loss_ema = (1.0 - 0.9) * rec_loss_vae + 0.9 * vae_loss_ema
            theta = geco_beta_update(theta, vae_loss_ema, g_goal, g_lr, speedup=2)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{config['num_epochs']}] Step [{i}] Loss: {loss.item():.4f}")
    
    # Save checkpoint
    if epoch % 4 == 0:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'AMPScaler': scaler.state_dict()
        }
        os.makedirs(config['save_path'], exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(config['save_path'], f"{config['train_id']}-epoch-{epoch}.pth.tar")
        )
        print(f"Checkpoint saved at epoch {epoch}")

print("Training complete!")
```

### Running Training

```bash
python run_training.py
```

---

## Testing/Inference

### Option 1: Using Locally Trained Model

```python
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology, filters
from torchio import transforms
from torch.cuda.amp import autocast

from ccevae import VAE
import torch.nn as nn

# Load model from checkpoint
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize model architecture
input_size = (1, 256, 256)
z_dim = 1024
model_feature_map_sizes = (16, 64, 256, 1024)

model = VAE(
    input_size=input_size,
    z_dim=z_dim,
    fmap_sizes=model_feature_map_sizes,
    conv_op=nn.Conv2d,
    tconv_op=nn.ConvTranspose2d,
    activation_op=torch.nn.PReLU
)
model.d = 2

# Load checkpoint
checkpoint = torch.load('checkpoints/ceVAE2D_brain-epoch-96.pth.tar', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# Load and preprocess test volume
def preprocess_volume(nifti_path):
    """Load and preprocess a NIfTI volume for testing."""
    vol = nib.load(nifti_path).get_fdata()
    vol = np.moveaxis(vol, 2, 0)  # Move slices to first dimension
    
    # Pad/crop to 256x256
    data_item = torch.tensor(vol).unsqueeze(dim=0)
    out = transforms.CropOrPad((vol.shape[0], 256, 256))(data_item)
    out = out.squeeze(dim=0).unsqueeze(dim=1)
    
    return out.float()

# Anomaly detection pipeline
def detect_anomalies(model, volume, device):
    """
    Run StRegA anomaly detection on a volume.
    
    Returns:
        anomaly_mask: Binary mask of detected anomalies
        diff_map: Continuous difference map
    """
    volume = volume.to(device)
    
    # Normalize
    volume = (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume))
    volume = torch.nan_to_num(volume, nan=0.0)
    
    with torch.no_grad():
        with autocast():
            reconstruction, _ = model(volume)
    
    reconstruction = reconstruction.float()
    
    # Calculate difference (reconstruction error)
    diff_mask = (reconstruction.cpu().numpy() - volume.cpu().numpy())
    
    # Post-processing
    # 1. Remove negative differences (we only care about reconstructing anomalies)
    m_diff_mask = diff_mask.copy()
    m_diff_mask[m_diff_mask < 0] = 0
    
    # 2. Manual thresholding (optional initial threshold)
    m_diff_mask[m_diff_mask > 0.2] = 1
    
    # 3. Otsu thresholding
    val = filters.threshold_otsu(m_diff_mask)
    thr = m_diff_mask > val
    thr[thr < 0] = 0
    
    # 4. Morphological opening to remove small false positives
    final = np.zeros_like(thr)
    for i in range(thr.shape[0]):
        final[i, 0] = morphology.area_opening(thr[i, 0], area_threshold=256)
    
    # Remove detections outside brain mask
    final[volume.cpu().numpy() == 0] = 0
    
    return final, diff_mask

# Example usage
volume = preprocess_volume('path/to/test_volume_segmented.nii.gz')
anomaly_mask, diff_map = detect_anomalies(model, volume, device)

print(f"Volume shape: {volume.shape}")
print(f"Anomaly mask shape: {anomaly_mask.shape}")
print(f"Number of anomalous voxels: {np.sum(anomaly_mask)}")
```

### Option 2: Using HuggingFace Pre-trained Model

```python
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology, filters
from torchio import transforms
from torch.cuda.amp import autocast
from transformers import AutoModel

# Load model from HuggingFace
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

modelHF = AutoModel.from_pretrained(
    "soumickmj/StRegA_cceVAE2D_Brain_MOOD_IXIT1_IXIT2_IXIPD",
    trust_remote_code=True
)
model = modelHF.model.to(device)
model.eval()

# Optional: Save HuggingFace model as local checkpoint
# torch.save(model, "checkpoints/brain_huggingface.ptrh")

# Load and preprocess test volume
def preprocess_volume(nifti_path):
    """Load and preprocess a NIfTI volume for testing."""
    vol = nib.load(nifti_path).get_fdata()
    vol = np.moveaxis(vol, 2, 0)
    
    data_item = torch.tensor(vol).unsqueeze(dim=0)
    out = transforms.CropOrPad((vol.shape[0], 256, 256))(data_item)
    out = out.squeeze(dim=0).unsqueeze(dim=1)
    
    return out.float()

# Anomaly detection function (same as above)
def detect_anomalies(model, volume, device):
    """Run StRegA anomaly detection."""
    volume = volume.to(device)
    volume = (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume))
    volume = torch.nan_to_num(volume, nan=0.0)
    
    with torch.no_grad():
        with autocast():
            reconstruction, _ = model(volume)
    
    reconstruction = reconstruction.float()
    diff_mask = (reconstruction.cpu().numpy() - volume.cpu().numpy())
    
    m_diff_mask = diff_mask.copy()
    m_diff_mask[m_diff_mask < 0] = 0
    m_diff_mask[m_diff_mask > 0.2] = 1
    
    val = filters.threshold_otsu(m_diff_mask)
    thr = m_diff_mask > val
    thr[thr < 0] = 0
    
    final = np.zeros_like(thr)
    for i in range(thr.shape[0]):
        final[i, 0] = morphology.area_opening(thr[i, 0], area_threshold=256)
    final[volume.cpu().numpy() == 0] = 0
    
    return final, diff_mask

# Example usage
volume = preprocess_volume('path/to/test_volume_segmented.nii.gz')
anomaly_mask, diff_map = detect_anomalies(model, volume, device)

print(f"Anomaly detection complete using HuggingFace model!")
print(f"Detected {np.sum(anomaly_mask)} anomalous voxels")
```

### Evaluation Metrics

```python
def dice_coefficient(true_mask, pred_mask, non_seg_score=1.0):
    """
    Compute Dice coefficient between two binary masks.
    
    Args:
        true_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
        non_seg_score: Score to return when both masks are empty
    
    Returns:
        Dice coefficient (0-1)
    """
    assert true_mask.shape == pred_mask.shape
    
    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score
    
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum

# Example evaluation
# dice = dice_coefficient(ground_truth_mask, anomaly_mask)
# print(f"Dice Score: {dice:.4f}")
```

---

## Paper Experiments

### Experiment 1: IXI Dataset Training

Training on IXI dataset with T1, T2, and PD-weighted images:

```python
# Dataset configuration
ixi_config = {
    't1_path': 'path/to/ixi_t1_segmented.hdf5',
    't2_path': 'path/to/ixi_t2_segmented.hdf5', 
    'pd_path': 'path/to/ixi_pd_segmented.hdf5',
    'train_indices': list(range(100, 350)),  # ~250 subjects
    'val_indices': list(range(30, 50)),       # ~20 subjects
}
```

### Experiment 2: MOOD Challenge Training

Training on MOOD (Medical Out-of-Distribution) T1 brain data:

```python
mood_config = {
    'data_path': 'path/to/mood_t1_seg.h5',
    'train_indices': list(range(100, 350)),
}
```

### Experiment 3: Combined Training (Master Model)

The HuggingFace "master" checkpoint was trained on combined data:

- MOOD T1
- IXI T1
- IXI T2  
- IXI PD

```python
# Combine datasets
from torch.utils.data import ConcatDataset

combined_trainset = ConcatDataset([
    ixi_t1_trainset,
    ixi_t2_trainset,
    ixi_pd_trainset,
    mood_trainset
])
```

### Experiment 4: BraTS Evaluation

Testing on BraTS dataset for tumor detection:

```python
# BraTS test data structure
brats_paths = {
    'non_seg': 'path/to/brats/non_seg/',      # Original images
    'seg': 'path/to/brats/seg/',               # FSL-segmented images  
    'mask': 'path/to/brats/mask/'              # Ground truth tumor masks
}

# Run evaluation on BraTS
import os

file_list = sorted(os.listdir(brats_paths['seg']))
dice_scores = []

for file in file_list:
    # Load data
    seg_vol = nib.load(os.path.join(brats_paths['seg'], file)).get_fdata()
    gt_mask = nib.load(os.path.join(brats_paths['mask'], file)).get_fdata()
    
    # Preprocess
    volume = preprocess_volume(os.path.join(brats_paths['seg'], file))
    
    # Detect anomalies
    anomaly_mask, _ = detect_anomalies(model, volume, device)
    
    # Calculate Dice
    gt_mask[gt_mask > 0] = 1
    dice = dice_coefficient(gt_mask, anomaly_mask)
    dice_scores.append(dice)
    print(f"{file}: Dice = {dice:.4f}")

print(f"\nMean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
```

---

## Additional Features in Repo

The following features are implemented in the repository but not detailed in the paper:

### 1. GMVAE (Gaussian Mixture VAE)
Located in `misc/gmvae.py`, implements a Gaussian Mixture VAE for clustering-based anomaly detection.

```python
from misc.gmvae import GMVAE, GMVAE_Trainer
# See misc/gmvae.py for implementation details
```

### 2. Skip-connection Autoencoder
Located in `misc/skipae.py` and `ceVae/skipae.py`, implements an autoencoder with skip connections.

```python
from misc.skipae import Skip_AE
# Alternative architecture with skip connections
```

### 3. Scale-space VAE (SSVAE)
Located in `misc/ssae.py`, implements scale-space decomposition for anomaly detection.

### 4. GECO (Generalized ELBO with Constrained Optimization)
Automatic balancing of reconstruction and KL losses:

```python
# Enable GECO in training
config['use_geco'] = True

# Parameters
g_goal = 0.1   # Target reconstruction error
g_lr = 1e-4    # GECO learning rate
```

### 5. 3D Model Support
The architecture supports both 2D and 3D models:

```python
# For 3D model
conv = nn.Conv3d
convt = nn.ConvTranspose3d
model.d = 3
```

### 6. Multiple Dataloader Variants
- `dataloaders/mood.py`: MOOD dataset loader
- `dataloaders/ixi.py`: IXI dataset loader
- `dataloaders/torchiowrap.py`: TorchIO wrapper for HDF5 data

---

## Features from Paper Not in Repo

The following aspects mentioned in the paper may require additional implementation:

### 1. Quantitative Evaluation Metrics
The paper reports multiple metrics that should be computed:

```python
def compute_metrics(pred_mask, true_mask):
    """Compute evaluation metrics from the paper."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    pred_flat = pred_mask.flatten().astype(bool)
    true_flat = true_mask.flatten().astype(bool)
    
    dice = dice_coefficient(true_mask, pred_mask)
    precision = precision_score(true_flat, pred_flat)
    recall = recall_score(true_flat, pred_flat)
    f1 = f1_score(true_flat, pred_flat)
    
    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### 2. Ablation Study Configurations
The paper includes ablation studies with different CE factors:

```python
# Ablation: CE factor variations
ce_factors_to_test = [0.0, 0.25, 0.5, 0.75, 1.0]

for ce_factor in ce_factors_to_test:
    config['ce_factor'] = ce_factor
    # Run training and evaluation
```

### 3. Cross-validation Setup
For robust evaluation:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(subject_ids)):
    # Train and evaluate on each fold
    pass
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size
   - Use gradient checkpointing
   - Reduce patches_per_volume

2. **NaN Loss Values**
   - Check data normalization
   - Lower learning rate
   - Add gradient clipping

3. **Poor Anomaly Detection**
   - Ensure FSL segmentation is correct
   - Adjust Otsu threshold
   - Tune morphological opening area_threshold

### Performance Tips

1. Use mixed precision training (enabled by default with GradScaler)
2. Increase num_workers for faster data loading
3. Use SSD storage for faster HDF5 access

---

## Citation

If you use this code, please cite:

```bibtex
@article{chatterjee2022strega,
  title={StRegA: Unsupervised Anomaly Detection in Brain MRIs using a Compact Context-encoding Variational Autoencoder},
  author={Chatterjee, Soumick and Sciarra, Alessandro and D{\"u}nnwald, Max and Tummala, Pavan and Agrawal, Shubham Kumar and Jauhari, Aishwarya and Kalra, Aman and Oeltze-Jafra, Steffen and Speck, Oliver and N{\"u}rnberger, Andreas},
  journal={Computers in Biology and Medicine},
  pages={106093},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2022.106093}
}
```

---

## Contact

For questions or issues:
- Email: soumick.chatterjee@ovgu.de or contact@soumick.com
- GitHub Issues: https://github.com/soumickmj/StRegA/issues
