#!/usr/bin/env python
"""
StRegA Inference Script

This script provides a complete inference pipeline for anomaly detection
using StRegA models. It supports both locally trained models and the
HuggingFace pre-trained model.

Usage:
    # Using HuggingFace model:
    python inference.py --input /path/to/volume.nii.gz --output /path/to/output/

    # Using local checkpoint:
    python inference.py --input /path/to/volume.nii.gz --output /path/to/output/ \
                        --checkpoint /path/to/checkpoint.pth.tar
                        
    # Using saved HuggingFace model:
    python inference.py --input /path/to/volume.nii.gz --output /path/to/output/ \
                        --checkpoint /path/to/brain.ptrh --checkpoint_format ptrh
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy import ndimage
from skimage import morphology, filters
from torch.cuda.amp import autocast
from torchio import transforms


def load_model_from_checkpoint(checkpoint_path, device, checkpoint_format='pth.tar'):
    """
    Load model from a local checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: torch device to load model on
        checkpoint_format: 'pth.tar' for training checkpoints, 'ptrh' for saved models
    
    Returns:
        Loaded model in eval mode
    """
    from ccevae import VAE
    
    if checkpoint_format == 'ptrh':
        # Directly saved model
        model = torch.load(checkpoint_path, map_location=device)
    else:
        # Training checkpoint with state_dict
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
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()
    return model


def load_model_from_huggingface(device):
    """
    Load pre-trained model from HuggingFace.
    
    Args:
        device: torch device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    from transformers import AutoModel
    
    print("Loading model from HuggingFace...")
    modelHF = AutoModel.from_pretrained(
        "soumickmj/StRegA_cceVAE2D_Brain_MOOD_IXIT1_IXIT2_IXIPD",
        trust_remote_code=True
    )
    model = modelHF.model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def preprocess_volume(nifti_path, target_size=(256, 256)):
    """
    Load and preprocess a NIfTI volume for inference.
    
    Args:
        nifti_path: Path to the NIfTI file
        target_size: Target spatial dimensions (H, W)
    
    Returns:
        Preprocessed tensor of shape (N_slices, 1, H, W)
    """
    # Load volume
    nii = nib.load(nifti_path)
    vol = nii.get_fdata()
    affine = nii.affine
    
    # Store original shape for later
    original_shape = vol.shape
    
    # Move slices to first dimension
    vol = np.moveaxis(vol, 2, 0)
    
    # Convert to tensor and add channel dimension
    data_item = torch.tensor(vol).unsqueeze(dim=0).float()
    
    # Crop or pad to target size
    target_shape = (vol.shape[0], target_size[0], target_size[1])
    out = transforms.CropOrPad(target_shape)(data_item)
    out = out.squeeze(dim=0).unsqueeze(dim=1)
    
    return out, affine, original_shape


def detect_anomalies(model, volume, device, area_threshold=256):
    """
    Run StRegA anomaly detection on a preprocessed volume.
    
    Args:
        model: Trained StRegA model
        volume: Preprocessed volume tensor (N_slices, 1, H, W)
        device: torch device
        area_threshold: Minimum area for morphological opening
    
    Returns:
        anomaly_mask: Binary mask of detected anomalies
        diff_map: Continuous difference map
        reconstruction: Model reconstruction
    """
    volume = volume.to(device)
    
    # Normalize to [0, 1]
    vol_min = torch.min(volume)
    vol_max = torch.max(volume)
    if vol_max > vol_min:
        volume_norm = (volume - vol_min) / (vol_max - vol_min)
    else:
        volume_norm = volume
    volume_norm = torch.nan_to_num(volume_norm, nan=0.0)
    
    # Run inference
    with torch.no_grad():
        with autocast():
            reconstruction, _ = model(volume_norm)
    
    reconstruction = reconstruction.float()
    
    # Calculate difference (reconstruction error)
    diff_map = (reconstruction.cpu().numpy() - volume_norm.cpu().numpy())
    
    # Post-processing pipeline
    # 1. Keep only positive differences (under-reconstruction indicates anomaly)
    m_diff_mask = diff_map.copy()
    m_diff_mask[m_diff_mask < 0] = 0
    
    # 2. Initial thresholding
    m_diff_mask[m_diff_mask > 0.2] = 1
    
    # 3. Otsu thresholding for adaptive binarization
    if m_diff_mask.max() > 0:
        val = filters.threshold_otsu(m_diff_mask)
        thr = m_diff_mask > val
    else:
        thr = m_diff_mask > 0
    thr = thr.astype(float)
    thr[thr < 0] = 0
    
    # 4. Morphological opening to remove small false positives
    final = np.zeros_like(thr)
    for i in range(thr.shape[0]):
        final[i, 0] = morphology.area_opening(
            thr[i, 0].astype(bool), 
            area_threshold=area_threshold
        ).astype(float)
    
    # 5. Remove detections outside brain mask
    brain_mask = volume_norm.cpu().numpy() > 0
    final[~brain_mask] = 0
    
    return final, diff_map, reconstruction.cpu().numpy()


def save_results(output_dir, anomaly_mask, diff_map, reconstruction, 
                 original_shape, affine, base_name):
    """
    Save inference results as NIfTI files.
    
    Args:
        output_dir: Output directory path
        anomaly_mask: Binary anomaly mask
        diff_map: Difference map
        reconstruction: Model reconstruction
        original_shape: Original volume shape
        affine: NIfTI affine matrix
        base_name: Base name for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove channel dimension and move slices back
    anomaly_mask = anomaly_mask.squeeze(1)
    diff_map = diff_map.squeeze(1)
    reconstruction = reconstruction.squeeze(1)
    
    # Move slices back to original position
    anomaly_mask = np.moveaxis(anomaly_mask, 0, 2)
    diff_map = np.moveaxis(diff_map, 0, 2)
    reconstruction = np.moveaxis(reconstruction, 0, 2)
    
    # Crop/pad back to original shape if needed
    # (simplified - actual implementation would need proper resampling)
    
    # Save anomaly mask
    anomaly_nii = nib.Nifti1Image(anomaly_mask.astype(np.float32), affine)
    nib.save(anomaly_nii, os.path.join(output_dir, f"{base_name}_anomaly_mask.nii.gz"))
    
    # Save difference map
    diff_nii = nib.Nifti1Image(diff_map.astype(np.float32), affine)
    nib.save(diff_nii, os.path.join(output_dir, f"{base_name}_diff_map.nii.gz"))
    
    # Save reconstruction
    recon_nii = nib.Nifti1Image(reconstruction.astype(np.float32), affine)
    nib.save(recon_nii, os.path.join(output_dir, f"{base_name}_reconstruction.nii.gz"))
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='StRegA Inference Script for Brain MRI Anomaly Detection'
    )
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True,
        help='Path to input NIfTI file (FSL-segmented brain MRI)'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--checkpoint', '-c', 
        type=str, 
        default=None,
        help='Path to model checkpoint (uses HuggingFace if not provided)'
    )
    parser.add_argument(
        '--checkpoint_format', 
        type=str, 
        default='pth.tar',
        choices=['pth.tar', 'ptrh'],
        help='Checkpoint format: pth.tar (training checkpoint) or ptrh (saved model)'
    )
    parser.add_argument(
        '--device', '-d', 
        type=str, 
        default='cuda:0',
        help='Device to run inference on (default: cuda:0)'
    )
    parser.add_argument(
        '--area_threshold', 
        type=int, 
        default=256,
        help='Minimum area for morphological opening (default: 256)'
    )
    parser.add_argument(
        '--save_huggingface', 
        type=str, 
        default=None,
        help='Path to save HuggingFace model as local checkpoint'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = load_model_from_checkpoint(
            args.checkpoint, 
            device, 
            args.checkpoint_format
        )
    else:
        model = load_model_from_huggingface(device)
        
        # Optionally save HuggingFace model locally
        if args.save_huggingface:
            print(f"Saving HuggingFace model to: {args.save_huggingface}")
            torch.save(model, args.save_huggingface)
    
    # Load and preprocess input
    print(f"Loading input: {args.input}")
    volume, affine, original_shape = preprocess_volume(args.input)
    print(f"Volume shape: {volume.shape}")
    
    # Run anomaly detection
    print("Running anomaly detection...")
    anomaly_mask, diff_map, reconstruction = detect_anomalies(
        model, volume, device, args.area_threshold
    )
    
    # Calculate statistics
    num_anomalous = np.sum(anomaly_mask > 0)
    total_voxels = np.prod(anomaly_mask.shape)
    anomaly_percentage = (num_anomalous / total_voxels) * 100
    
    print(f"Detection complete!")
    print(f"  Anomalous voxels: {num_anomalous}")
    print(f"  Total voxels: {total_voxels}")
    print(f"  Anomaly percentage: {anomaly_percentage:.4f}%")
    
    # Save results
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    base_name = base_name.replace('.nii', '')  # Handle .nii.gz
    save_results(
        args.output, 
        anomaly_mask, 
        diff_map, 
        reconstruction,
        original_shape,
        affine,
        base_name
    )
    
    print("Inference complete!")


if __name__ == '__main__':
    main()
