#!/usr/bin/env python3
"""Evaluate trained model and compute real-world metrics."""

import argparse
import torch
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.gat import create_model


@torch.no_grad()
def evaluate(model, loader, device, scaler_params):
    """Evaluate model and denormalize predictions."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.global_features, batch.batch)
        
        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
    
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Denormalize (reverse z-score)
    # Find time_elapsed in scaler params
    target_idx = scaler_params['columns'].index('time_elapsed')
    mean = scaler_params['mean'][target_idx]
    std = scaler_params['std'][target_idx]
    
    preds_real = preds * std + mean
    targets_real = targets * std + mean
    
    # Compute metrics
    mse = np.mean((preds_real - targets_real) ** 2)
    mae = np.mean(np.abs(preds_real - targets_real))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds_real - targets_real) / (targets_real + 1e-8))) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'preds': preds_real,
        'targets': targets_real
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='best.pth')
    args = parser.parse_args()
    
    exp_dir = Path('experiments/checkpoints') / args.experiment
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint with proper device mapping
    checkpoint = torch.load(
        exp_dir / args.checkpoint,
        map_location=device  # This fixes the error
    )
    config = checkpoint['config']
    
    # Load scaler params
    with open(config['paths']['metadata'], 'r') as f:
        metadata = json.load(f)
    
    # Setup dataloaders
    print("\nLoading test data...")
    _, _, test_loader, _ = create_dataloaders(
        train_path=config['paths']['train'],
        val_path=config['paths']['val'],
        test_path=config['paths']['test'],
        batch_size=config['training']['batch_size'],
        num_workers=0,
        metadata_path=config['paths']['metadata']
    )
    
    # Load model
    print("Loading model...")
    model = create_model(config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    print("Evaluating...")
    results = evaluate(model, test_loader, device, metadata['scaler_params'])
    
    print(f"\n{'='*60}")
    print(f"Test Set Results (Real Scale - Time in Seconds)")
    print(f"{'='*60}")
    print(f"  MSE:  {results['mse']:.4f} s²")
    print(f"  RMSE: {results['rmse']:.4f} s")
    print(f"  MAE:  {results['mae']:.4f} s")
    print(f"  MAPE: {results['mape']:.2f}%")
    
    # Sample predictions
    print(f"\nSample Predictions (first 10):")
    print(f"  {'Predicted':>12s}  {'Actual':>12s}  {'Error':>12s}")
    for i in range(min(10, len(results['preds']))):
        pred = results['preds'][i]
        target = results['targets'][i]
        error = pred - target
        print(f"  {pred:12.4f}  {target:12.4f}  {error:12.4f}")
    
    # Save results
    save_dict = {k: v for k, v in results.items() if k not in ['preds', 'targets']}
    with open(exp_dir / 'test_results.json', 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {exp_dir}/test_results.json")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()