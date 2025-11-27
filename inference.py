#!/usr/bin/env python
"""
StRegA Inference Script

This script provides a complete inference pipeline for anomaly detection
using StRegA models. It supports both locally trained models and the
HuggingFace pre-trained model.

Supported input formats:
    1. Single NIfTI file (.nii or .nii.gz)
    2. Directory containing NIfTI files
    3. HDF5 file (same format as training dataset)

Usage:
    # Single NIfTI file with HuggingFace model:
    python inference.py --input /path/to/volume.nii.gz --output /path/to/output/

    # Directory of NIfTI files:
    python inference.py --input /path/to/nifti_folder/ --output /path/to/output/

    # HDF5 file (training dataset format):
    python inference.py --input /path/to/data.h5 --output /path/to/output/ --input_format h5

    # HDF5 with region key (MOOD format):
    python inference.py --input /path/to/mood.h5 --output /path/to/output/ --input_format h5 --h5_region brain

    # Using local checkpoint:
    python inference.py --input /path/to/volume.nii.gz --output /path/to/output/ \
                        --checkpoint /path/to/checkpoint.pth.tar
                        
    # Using saved HuggingFace model:
    python inference.py --input /path/to/volume.nii.gz --output /path/to/output/ \
                        --checkpoint /path/to/brain.ptrh --checkpoint_format ptrh
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import h5py as h5
from skimage import morphology, filters
from torch.cuda.amp import autocast
from torchio import transforms

# Default threshold for initial anomaly detection (values above this are considered anomalies)
DEFAULT_ANOMALY_THRESHOLD = 0.2


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


def preprocess_volume_from_array(vol, target_size=(256, 256)):
    """
    Preprocess a numpy array volume for inference.
    
    Args:
        vol: numpy array of shape (H, W, D) or (D, H, W)
        target_size: Target spatial dimensions (H, W)
    
    Returns:
        Preprocessed tensor of shape (N_slices, 1, H, W)
        original_shape: Original volume shape
    """
    original_shape = vol.shape
    
    # Ensure slices are in first dimension
    # Assume the smallest dimension is the slice dimension if ambiguous
    if vol.ndim == 3:
        # If last dimension is smallest, assume it's (H, W, D) format
        if vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
            vol = np.moveaxis(vol, 2, 0)
        # Otherwise assume already in (D, H, W) format
    
    # Convert to tensor and add channel dimension
    data_item = torch.tensor(vol).unsqueeze(dim=0).float()
    
    # Crop or pad to target size
    target_shape = (vol.shape[0], target_size[0], target_size[1])
    out = transforms.CropOrPad(target_shape)(data_item)
    out = out.squeeze(dim=0).unsqueeze(dim=1)
    
    return out, original_shape


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


def detect_anomalies(model, volume, device, area_threshold=256, 
                     anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD):
    """
    Run StRegA anomaly detection on a preprocessed volume.
    
    Args:
        model: Trained StRegA model
        volume: Preprocessed volume tensor (N_slices, 1, H, W)
        device: torch device
        area_threshold: Minimum area for morphological opening
        anomaly_threshold: Initial threshold for anomaly detection (default: 0.2)
    
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
    
    # 2. Initial thresholding using configurable threshold
    m_diff_mask[m_diff_mask > anomaly_threshold] = 1
    
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
    
    Note:
        The output files are saved in the processed coordinate space (256x256 slices).
        If the original volume had different dimensions, the outputs will not perfectly
        align with the original. For production use, consider implementing proper
        resampling to match original dimensions.
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


def save_results_to_h5(h5_file, anomaly_mask, diff_map, reconstruction, 
                       original_shape, subject_key):
    """
    Save inference results to an HDF5 file.
    
    Args:
        h5_file: Open HDF5 file object for writing
        anomaly_mask: Binary anomaly mask
        diff_map: Difference map
        reconstruction: Model reconstruction
        original_shape: Original volume shape
        subject_key: Key to use for this subject in the HDF5 file
    """
    # Remove channel dimension and move slices back
    anomaly_mask = anomaly_mask.squeeze(1)
    diff_map = diff_map.squeeze(1)
    reconstruction = reconstruction.squeeze(1)
    
    # Move slices back to original position
    anomaly_mask = np.moveaxis(anomaly_mask, 0, 2)
    diff_map = np.moveaxis(diff_map, 0, 2)
    reconstruction = np.moveaxis(reconstruction, 0, 2)
    
    # Create group for this subject
    if subject_key in h5_file:
        del h5_file[subject_key]
    grp = h5_file.create_group(subject_key)
    
    # Save arrays
    grp.create_dataset('anomaly_mask', data=anomaly_mask.astype(np.float32))
    grp.create_dataset('diff_map', data=diff_map.astype(np.float32))
    grp.create_dataset('reconstruction', data=reconstruction.astype(np.float32))


def load_h5_subjects(h5_path, region=None, indices=None):
    """
    Load subject data from an HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        region: Optional region key (e.g., 'brain' for MOOD format)
        indices: Optional list of indices to load
    
    Yields:
        Tuple of (subject_key, volume_data)
    
    Note:
        When indices are provided, the function tries multiple key formats:
        - Zero-padded with 5 digits (e.g., '00000', '00001') for MOOD/IXI format
        - Plain string (e.g., '0', '1') as fallback
    """
    with h5.File(h5_path, 'r', swmr=True) as h5_file:
        if region and region in h5_file:
            # MOOD format: data under region key
            base = h5_file[region]
        else:
            # IXI format: data directly under root
            base = h5_file
        
        # Get all keys or filter by indices
        if indices:
            keys = []
            for i in indices:
                # Try zero-padded format first (MOOD/IXI format)
                key_padded = str(i).zfill(5)
                if key_padded in base:
                    keys.append(key_padded)
                # Try plain string format as fallback
                elif str(i) in base:
                    keys.append(str(i))
        else:
            keys = sorted([k for k in base.keys() if not k.startswith('_')])
        
        for key in keys:
            try:
                data = base[key][()]
                yield key, data
            except Exception as e:
                print(f"Warning: Could not load {key}: {e}")
                continue


def get_nifti_files(input_path):
    """
    Get list of NIfTI files from a path.
    
    Args:
        input_path: Path to a single file or directory
    
    Returns:
        List of NIfTI file paths
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        nifti_files = []
        for ext in ['*.nii', '*.nii.gz']:
            nifti_files.extend(glob.glob(os.path.join(input_path, ext)))
            nifti_files.extend(glob.glob(os.path.join(input_path, '**', ext), recursive=True))
        return sorted(list(set(nifti_files)))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def process_single_volume(model, volume, device, area_threshold, anomaly_threshold):
    """
    Process a single volume and return results with statistics.
    
    Args:
        model: Loaded model
        volume: Preprocessed volume tensor
        device: torch device
        area_threshold: Minimum area for morphological opening
        anomaly_threshold: Initial threshold for anomaly detection
    
    Returns:
        Tuple of (anomaly_mask, diff_map, reconstruction, stats_dict)
    """
    anomaly_mask, diff_map, reconstruction = detect_anomalies(
        model, volume, device, area_threshold, anomaly_threshold
    )
    
    num_anomalous = np.sum(anomaly_mask > 0)
    total_voxels = np.prod(anomaly_mask.shape)
    anomaly_percentage = (num_anomalous / total_voxels) * 100
    
    stats = {
        'anomalous_voxels': int(num_anomalous),
        'total_voxels': int(total_voxels),
        'anomaly_percentage': float(anomaly_percentage)
    }
    
    return anomaly_mask, diff_map, reconstruction, stats


def main():
    parser = argparse.ArgumentParser(
        description='StRegA Inference Script for Brain MRI Anomaly Detection'
    )
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True,
        help='Path to input: NIfTI file, directory of NIfTI files, or HDF5 file'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        required=True,
        help='Output directory for results (or .h5 file path for HDF5 output)'
    )
    parser.add_argument(
        '--input_format',
        type=str,
        default='auto',
        choices=['auto', 'nifti', 'h5'],
        help='Input format: auto (detect from extension), nifti, or h5 (default: auto)'
    )
    parser.add_argument(
        '--h5_region',
        type=str,
        default=None,
        help='Region key for HDF5 file (e.g., "brain" for MOOD format)'
    )
    parser.add_argument(
        '--h5_indices',
        type=str,
        default=None,
        help='Comma-separated indices or range (e.g., "0,1,2" or "0-100") for HDF5 input'
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
        '--anomaly_threshold',
        type=float,
        default=DEFAULT_ANOMALY_THRESHOLD,
        help=f'Initial threshold for anomaly detection (default: {DEFAULT_ANOMALY_THRESHOLD})'
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
    
    # Detect input format
    input_format = args.input_format
    if input_format == 'auto':
        if args.input.endswith('.h5') or args.input.endswith('.hdf5'):
            input_format = 'h5'
        else:
            input_format = 'nifti'
    
    # Parse HDF5 indices if provided
    h5_indices = None
    if args.h5_indices:
        try:
            # Check if it's a range (exactly one hyphen with numbers on both sides)
            if args.h5_indices.count('-') == 1 and not args.h5_indices.startswith('-'):
                parts = args.h5_indices.split('-')
                if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    h5_indices = list(range(start, end + 1))
                else:
                    raise ValueError(f"Invalid range format: {args.h5_indices}")
            else:
                # Comma-separated list
                h5_indices = [int(x.strip()) for x in args.h5_indices.split(',') if x.strip()]
        except ValueError as e:
            print(f"Error parsing --h5_indices '{args.h5_indices}': {e}")
            print("Expected format: '0,1,2' (comma-separated) or '0-100' (range)")
            return
    
    # Process based on input format
    if input_format == 'h5':
        print(f"Processing HDF5 file: {args.input}")
        
        # Determine output format
        if args.output.endswith('.h5') or args.output.endswith('.hdf5'):
            # Output to HDF5
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            output_h5 = True
        else:
            # Output to directory with NIfTI files
            os.makedirs(args.output, exist_ok=True)
            output_h5 = False
        
        all_stats = []
        h5_out = None
        try:
            if output_h5:
                h5_out = h5.File(args.output, 'w')
            
            for subject_key, vol_data in load_h5_subjects(args.input, args.h5_region, h5_indices):
                print(f"  Processing subject: {subject_key}")
                
                # Preprocess volume
                volume, original_shape = preprocess_volume_from_array(vol_data)
                
                # Run detection
                anomaly_mask, diff_map, reconstruction, stats = process_single_volume(
                    model, volume, device, args.area_threshold, args.anomaly_threshold
                )
                stats['subject'] = subject_key
                all_stats.append(stats)
                
                print(f"    Anomalous voxels: {stats['anomalous_voxels']}")
                print(f"    Anomaly percentage: {stats['anomaly_percentage']:.4f}%")
                
                # Save results
                if output_h5:
                    save_results_to_h5(h5_out, anomaly_mask, diff_map, reconstruction,
                                       original_shape, subject_key)
                else:
                    # Save as NIfTI with identity affine
                    # Note: HDF5 files don't contain spatial orientation info,
                    # so an identity affine is used. This may affect spatial analysis.
                    affine = np.eye(4)
                    save_results(args.output, anomaly_mask, diff_map, reconstruction,
                                original_shape, affine, subject_key)
        finally:
            if h5_out is not None:
                h5_out.close()
        
        # Print summary
        print("\n" + "="*50)
        print("Summary Statistics")
        print("="*50)
        total_subjects = len(all_stats)
        avg_anomaly_pct = np.mean([s['anomaly_percentage'] for s in all_stats])
        print(f"Total subjects processed: {total_subjects}")
        print(f"Average anomaly percentage: {avg_anomaly_pct:.4f}%")
        
    else:
        # NIfTI input (single file or directory)
        nifti_files = get_nifti_files(args.input)
        
        if len(nifti_files) == 0:
            print(f"Error: No NIfTI files found in {args.input}")
            return
        
        print(f"Found {len(nifti_files)} NIfTI file(s) to process")
        os.makedirs(args.output, exist_ok=True)
        
        all_stats = []
        for nifti_path in nifti_files:
            print(f"\nProcessing: {nifti_path}")
            
            # Load and preprocess
            volume, affine, original_shape = preprocess_volume(nifti_path)
            print(f"  Volume shape: {volume.shape}")
            
            # Run detection
            anomaly_mask, diff_map, reconstruction, stats = process_single_volume(
                model, volume, device, args.area_threshold, args.anomaly_threshold
            )
            stats['file'] = os.path.basename(nifti_path)
            all_stats.append(stats)
            
            print(f"  Anomalous voxels: {stats['anomalous_voxels']}")
            print(f"  Anomaly percentage: {stats['anomaly_percentage']:.4f}%")
            
            # Save results
            base_name = os.path.splitext(os.path.basename(nifti_path))[0]
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
        
        # Print summary for multiple files
        if len(nifti_files) > 1:
            print("\n" + "="*50)
            print("Summary Statistics")
            print("="*50)
            total_files = len(all_stats)
            avg_anomaly_pct = np.mean([s['anomaly_percentage'] for s in all_stats])
            print(f"Total files processed: {total_files}")
            print(f"Average anomaly percentage: {avg_anomaly_pct:.4f}%")
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
