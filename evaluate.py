#!/usr/bin/env python
"""
StRegA Evaluation Script

This script evaluates StRegA anomaly detection results against ground truth masks
and computes metrics including Dice coefficient, precision, recall, and F1 score.

Usage:
    python evaluate.py --predictions /path/to/predictions/ --ground_truth /path/to/gt/ \
                       --output results.csv
"""

import argparse
import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob


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
    assert true_mask.shape == pred_mask.shape, \
        f"Shape mismatch: {true_mask.shape} vs {pred_mask.shape}"
    
    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score
    
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum


def precision_score(true_mask, pred_mask):
    """
    Compute precision: TP / (TP + FP)
    
    Args:
        true_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
    
    Returns:
        Precision score (0-1)
    
    Edge cases:
        - If no predictions are made and no positives exist: returns 1.0 (perfect precision)
        - If no predictions are made but positives exist: returns 0.0 (no true positives found)
    """
    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    
    true_positives = np.sum(np.logical_and(true_mask, pred_mask))
    predicted_positives = np.sum(pred_mask)
    
    if predicted_positives == 0:
        # No predictions made - return 1.0 if nothing to detect, 0.0 otherwise
        return 1.0 if np.sum(true_mask) == 0 else 0.0
    
    return true_positives / predicted_positives


def recall_score(true_mask, pred_mask):
    """
    Compute recall (sensitivity): TP / (TP + FN)
    
    Args:
        true_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
    
    Returns:
        Recall score (0-1)
    
    Edge cases:
        - If no actual positives exist and no predictions made: returns 1.0 (perfect recall)
        - If no actual positives exist but predictions made: returns 0.0 (false positives)
    """
    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    
    true_positives = np.sum(np.logical_and(true_mask, pred_mask))
    actual_positives = np.sum(true_mask)
    
    if actual_positives == 0:
        # No actual positives - return 1.0 if no predictions, 0.0 otherwise (false positives)
        return 1.0 if np.sum(pred_mask) == 0 else 0.0
    
    return true_positives / actual_positives


def f1_score(true_mask, pred_mask):
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall)
    
    Args:
        true_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
    
    Returns:
        F1 score (0-1)
    """
    prec = precision_score(true_mask, pred_mask)
    rec = recall_score(true_mask, pred_mask)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def specificity_score(true_mask, pred_mask):
    """
    Compute specificity: TN / (TN + FP)
    
    Args:
        true_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
    
    Returns:
        Specificity score (0-1)
    """
    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    
    true_negatives = np.sum(np.logical_and(~true_mask, ~pred_mask))
    actual_negatives = np.sum(~true_mask)
    
    if actual_negatives == 0:
        return 1.0
    
    return true_negatives / actual_negatives


def compute_all_metrics(true_mask, pred_mask):
    """
    Compute all evaluation metrics.
    
    Args:
        true_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
    
    Returns:
        Dictionary of metrics
    """
    return {
        'dice': dice_coefficient(true_mask, pred_mask),
        'precision': precision_score(true_mask, pred_mask),
        'recall': recall_score(true_mask, pred_mask),
        'f1': f1_score(true_mask, pred_mask),
        'specificity': specificity_score(true_mask, pred_mask),
        'gt_volume': np.sum(true_mask.astype(bool)),
        'pred_volume': np.sum(pred_mask.astype(bool))
    }


def load_nifti_mask(path, binarize=True, threshold=0):
    """
    Load a NIfTI mask file.
    
    Args:
        path: Path to NIfTI file
        binarize: Whether to binarize the mask
        threshold: Threshold for binarization
    
    Returns:
        NumPy array of the mask
    """
    nii = nib.load(path)
    mask = nii.get_fdata()
    
    if binarize:
        mask = (mask > threshold).astype(np.float32)
    
    return mask


def find_matching_files(pred_dir, gt_dir, pred_suffix='_anomaly_mask.nii.gz', 
                        gt_suffix='.nii.gz'):
    """
    Find matching prediction and ground truth file pairs.
    
    Args:
        pred_dir: Directory containing predictions
        gt_dir: Directory containing ground truth
        pred_suffix: Suffix for prediction files
        gt_suffix: Suffix for ground truth files
    
    Returns:
        List of (pred_path, gt_path, subject_id) tuples
    """
    pred_files = glob(os.path.join(pred_dir, f'*{pred_suffix}'))
    matches = []
    
    for pred_path in pred_files:
        # Extract subject ID by removing suffix and directory
        base_name = os.path.basename(pred_path)
        subject_id = base_name.replace(pred_suffix, '')
        
        # Look for matching ground truth
        gt_path = os.path.join(gt_dir, f'{subject_id}{gt_suffix}')
        
        if os.path.exists(gt_path):
            matches.append((pred_path, gt_path, subject_id))
        else:
            print(f"Warning: No ground truth found for {subject_id}")
    
    return matches


def main():
    parser = argparse.ArgumentParser(
        description='StRegA Evaluation Script'
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Directory containing prediction masks'
    )
    parser.add_argument(
        '--ground_truth', '-g',
        type=str,
        required=True,
        help='Directory containing ground truth masks'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results.csv',
        help='Output CSV file path (default: evaluation_results.csv)'
    )
    parser.add_argument(
        '--pred_suffix',
        type=str,
        default='_anomaly_mask.nii.gz',
        help='Suffix for prediction files (default: _anomaly_mask.nii.gz)'
    )
    parser.add_argument(
        '--gt_suffix',
        type=str,
        default='.nii.gz',
        help='Suffix for ground truth files (default: .nii.gz)'
    )
    parser.add_argument(
        '--binarize_gt',
        action='store_true',
        help='Binarize ground truth masks (for multi-class labels)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print per-subject results'
    )
    
    args = parser.parse_args()
    
    # Find matching files
    matches = find_matching_files(
        args.predictions, 
        args.ground_truth,
        args.pred_suffix,
        args.gt_suffix
    )
    
    if len(matches) == 0:
        print("Error: No matching file pairs found!")
        return
    
    print(f"Found {len(matches)} matching file pairs")
    
    # Evaluate each pair
    results = []
    
    for pred_path, gt_path, subject_id in matches:
        if args.verbose:
            print(f"\nEvaluating: {subject_id}")
        
        # Load masks
        pred_mask = load_nifti_mask(pred_path, binarize=True)
        gt_mask = load_nifti_mask(gt_path, binarize=args.binarize_gt)
        
        # Handle shape mismatches
        if pred_mask.shape != gt_mask.shape:
            print(f"  Warning: Shape mismatch for {subject_id}")
            print(f"    Prediction: {pred_mask.shape}")
            print(f"    Ground truth: {gt_mask.shape}")
            continue
        
        # Compute metrics
        metrics = compute_all_metrics(gt_mask, pred_mask)
        metrics['subject_id'] = subject_id
        results.append(metrics)
        
        if args.verbose:
            print(f"  Dice: {metrics['dice']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['subject_id', 'dice', 'precision', 'recall', 'f1', 
            'specificity', 'gt_volume', 'pred_volume']
    df = df[cols]
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    
    for metric in ['dice', 'precision', 'recall', 'f1', 'specificity']:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        print(f"{metric.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f} "
              f"(min: {min_val:.4f}, max: {max_val:.4f})")
    
    print("="*50)
    print(f"Number of subjects evaluated: {len(df)}")


if __name__ == '__main__':
    main()
