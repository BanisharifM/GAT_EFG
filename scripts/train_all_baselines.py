#!/usr/bin/env python3
"""
Comprehensive baseline comparison script.
Trains all methods: LR, RF, MLP, GCN, GAT (with/without CFG).
"""

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.gat import create_model
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleMLP(nn.Module):
    """MLP baseline - flattened features."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, edge_index, global_features, batch):
        batch_size = global_features.shape[0]
        x_flat = x.view(batch_size, -1)
        features = torch.cat([x_flat, global_features], dim=1)
        return self.model(features)


class SimpleGCN(nn.Module):
    """GCN baseline."""
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim=128):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, edge_index, global_features, batch):
        x = self.node_encoder(x)
        x = torch.relu(self.gcn1(x, edge_index))
        x = torch.relu(self.gcn2(x, edge_index))
        x = self.gcn3(x, edge_index)
        
        graph_repr = global_mean_pool(x, batch)
        global_repr = self.global_encoder(global_features)
        
        fused = torch.cat([graph_repr, global_repr], dim=1)
        return self.predictor(fused)


def prepare_sklearn_data(loader):
    """Extract flattened features for sklearn models."""
    X_list, y_list = [], []
    
    for batch in loader:
        batch_size = batch.global_features.shape[0]
        num_nodes = batch.x.shape[0] // batch_size
        
        # Flatten node features
        x_flat = batch.x.view(batch_size, -1)
        
        # Concatenate with global features
        features = torch.cat([x_flat, batch.global_features], dim=1)
        
        X_list.append(features.cpu().numpy())
        y_list.append(batch.y.cpu().numpy())
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list).flatten()
    
    return X, y


def train_sklearn_model(model, train_loader, val_loader):
    """Train sklearn model."""
    X_train, y_train = prepare_sklearn_data(train_loader)
    X_val, y_val = prepare_sklearn_data(val_loader)
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    return {
        'train_mse': float(train_mse),
        'val_mse': float(val_mse),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'train_time': train_time
    }


def train_pytorch_model(model, train_loader, val_loader, device, epochs=50):
    """Train PyTorch model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    start = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.global_features, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.global_features, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    train_time = time.time() - start
    
    # Final metrics
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    val_mae = mean_absolute_error(targets, preds)
    val_r2 = r2_score(targets, preds)
    
    return {
        'train_mse': float(train_loss),
        'val_mse': float(best_val_loss),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'train_time': train_time,
        'epochs': epoch + 1
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    results = {}
    
    # Run for both with/without CFG
    for use_cfg in [False, True]:
        cfg_suffix = "_with_cfg" if use_cfg else ""
        print(f"\n{'='*70}")
        print(f"Running experiments: {'WITH CFG FEATURES' if use_cfg else 'WITHOUT CFG FEATURES'}")
        print(f"{'='*70}\n")
        
        # Load data
        if use_cfg:
            train_path = 'data/splits/train_log_with_cfg.csv'
            val_path = 'data/splits/val_log_with_cfg.csv'
            test_path = 'data/splits/test_log_with_cfg.csv'
            metadata_path = 'data/processed/scaler_params_log_with_cfg.json'
        else:
            train_path = 'data/splits/train_log.csv'
            val_path = 'data/splits/val_log.csv'
            test_path = 'data/splits/test_log.csv'
            metadata_path = 'data/processed/scaler_params_log.json'
        
        train_loader, val_loader, test_loader, (node_dim, global_dim) = create_dataloaders(
            train_path, val_path, test_path,
            batch_size=32, num_workers=4,
            metadata_path=metadata_path,
            target_column='time_elapsed_log',
            use_cfg_features=use_cfg
        )
        
        input_dim = node_dim * 8 + global_dim
        
        # 1. Linear Regression
        print(f"Training Linear Regression{cfg_suffix}...")
        lr = LinearRegression()
        results[f'linear_regression{cfg_suffix}'] = train_sklearn_model(lr, train_loader, val_loader)
        print(f"  Val MSE: {results[f'linear_regression{cfg_suffix}']['val_mse']:.6f}")
        
        # 2. Random Forest
        print(f"Training Random Forest{cfg_suffix}...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        results[f'random_forest{cfg_suffix}'] = train_sklearn_model(rf, train_loader, val_loader)
        print(f"  Val MSE: {results[f'random_forest{cfg_suffix}']['val_mse']:.6f}")
        
        # 3. MLP
        print(f"Training MLP{cfg_suffix}...")
        mlp = SimpleMLP(input_dim, hidden_dims=[256, 128, 64])
        results[f'mlp{cfg_suffix}'] = train_pytorch_model(mlp, train_loader, val_loader, device)
        print(f"  Val MSE: {results[f'mlp{cfg_suffix}']['val_mse']:.6f}")
        
        # 4. GCN
        print(f"Training GCN{cfg_suffix}...")
        gcn = SimpleGCN(node_dim, global_dim, hidden_dim=128)
        results[f'gcn{cfg_suffix}'] = train_pytorch_model(gcn, train_loader, val_loader, device)
        print(f"  Val MSE: {results[f'gcn{cfg_suffix}']['val_mse']:.6f}")
        
        # 5. GAT
        print(f"Training GAT{cfg_suffix}...")
        gat = create_model({
            'node_feature_dim': node_dim,
            'global_feature_dim': global_dim,
            'hidden_dim': 128,
            'num_gat_layers': 3,
            'num_heads': 4,
            'dropout': 0.2,
            'pooling': 'both'
        })
        results[f'gat{cfg_suffix}'] = train_pytorch_model(gat, train_loader, val_loader, device)
        print(f"  Val MSE: {results[f'gat{cfg_suffix}']['val_mse']:.6f}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    print(f"{'Model':<25} {'Val MSE':>12} {'Val MAE':>12} {'Val R²':>12} {'Time (s)':>12} {'Epochs':>10}")
    print(f"{'-'*100}")
    
    for model_name, metrics in sorted(results.items()):
        epochs_str = str(metrics.get('epochs', 'N/A'))
        print(f"{model_name:<25} {metrics['val_mse']:12.6f} {metrics['val_mae']:12.4f} "
              f"{metrics['val_r2']:12.4f} {metrics['train_time']:12.2f} {epochs_str:>10}")
    
    # Save results
    output_file = 'experiments/baseline_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}\n")


if __name__ == '__main__':
    main()