"""
PyTorch Geometric Dataset for GAT-based performance prediction.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Optional
import json


class CoreGraphDataset(Dataset):
    """Dataset that creates graph representation of core configurations."""
    
    def __init__(self, 
                csv_path: str,
                metadata_path: Optional[str] = None,
                num_cores: int = 8,
                target_column: str = 'time_elapsed'):
        """Initialize dataset."""
        self.num_cores = num_cores
        self.df = pd.read_csv(csv_path)
        
        # Load metadata if provided
        self.metadata = None
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Define feature columns
        self.node_feature_cols = self._get_node_feature_columns()
        self.global_feature_cols = self._get_global_feature_columns()
        self.target_col = target_column  # Use parameter instead of hardcoded
        
        # Create edge index (fully connected graph)
        self.edge_index = self._create_edge_index()
        
        print(f"Dataset initialized:")
        print(f"  Samples: {len(self.df)}")
        print(f"  Node features per core: {len(self.node_feature_cols) // num_cores}")
        print(f"  Global features: {len(self.global_feature_cols)}")
        print(f"  Target: {self.target_col}")
        print(f"  Graph: {num_cores} nodes, fully connected")
    
    def _get_node_feature_columns(self) -> List[str]:
        """Get node feature column names."""
        node_features = []
        for i in range(self.num_cores):
            node_features.append(f'core_active_{i}')
        for i in range(self.num_cores):
            node_features.append(f'freq_lvl_{i}')
        for i in range(self.num_cores):
            node_features.append(f'freq_hz_{i}')
        return node_features
    
    def _get_global_feature_columns(self) -> List[str]:
        """Get global feature column names."""
        return [
            'run_mode',
            'benchmark',
            'platform',
            'frequency_index',
            'num_cores',
            'cores_str',
            'avg_temp_before',
            'thermal_init_mean'
        ]
    
    def _create_edge_index(self) -> torch.Tensor:
        """Create fully connected edge index."""
        edges = []
        for i in range(self.num_cores):
            for j in range(self.num_cores):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _create_node_features(self, row: pd.Series) -> torch.Tensor:
        """Create node feature matrix [num_nodes, 3]."""
        node_features = []
        for i in range(self.num_cores):
            core_active = row[f'core_active_{i}']
            freq_lvl = row[f'freq_lvl_{i}']
            freq_hz = row[f'freq_hz_{i}']
            node_features.append([core_active, freq_lvl, freq_hz])
        return torch.tensor(node_features, dtype=torch.float32)
    
    def _create_global_features(self, row: pd.Series) -> torch.Tensor:
        """Create global feature vector."""
        global_vals = [row[col] for col in self.global_feature_cols]
        return torch.tensor(global_vals, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Data:
        """Get a single graph sample."""
        row = self.df.iloc[idx]
        
        # Create graph components
        x = self._create_node_features(row)
        global_feat = self._create_global_features(row)
        y = torch.tensor([row[self.target_col]], dtype=torch.float32)
        
        # Store global features with special handling
        data = Data(
            x=x,
            edge_index=self.edge_index,
            y=y
        )
        
        # Add global features as a 2D tensor [1, feature_dim]
        data.global_features = global_feat.unsqueeze(0)
        
        return data
    
    def get_feature_dims(self) -> Tuple[int, int]:
        """Get feature dimensions."""
        node_feature_dim = 3
        global_feature_dim = len(self.global_feature_cols)
        return node_feature_dim, global_feature_dim


# Custom collate function to handle global features correctly
def collate_fn(data_list):
    """Custom collate to properly batch global features."""
    # Use PyG's default batching for graph components
    batch = Batch.from_data_list(data_list)
    
    # Manually stack global features [batch_size, global_dim]
    global_features = torch.cat([d.global_features for d in data_list], dim=0)
    batch.global_features = global_features
    
    return batch


def create_dataloaders(train_path: str,
                       val_path: str,
                       test_path: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       metadata_path: Optional[str] = None,
                       target_column: str = 'time_elapsed') -> Tuple:  # Add parameter
    """Create DataLoaders with custom collate function."""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CoreGraphDataset(train_path, metadata_path, target_column=target_column)
    val_dataset = CoreGraphDataset(val_path, metadata_path, target_column=target_column)
    test_dataset = CoreGraphDataset(test_path, metadata_path, target_column=target_column)
    
    
    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Get feature dimensions
    feature_dims = train_dataset.get_feature_dims()
    
    print("\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Feature dims: node={feature_dims[0]}, global={feature_dims[1]}")
    
    return train_loader, val_loader, test_loader, feature_dims


if __name__ == '__main__':
    print("Testing CoreGraphDataset...")
    
    dataset = CoreGraphDataset(
        csv_path='data/splits/train.csv',
        metadata_path='data/processed/merged_metadata.json'
    )
    
    sample = dataset[0]
    
    print("\nSample graph structure:")
    print(f"  Node features (x): {sample.x.shape}")
    print(f"  Edge index: {sample.edge_index.shape}")
    print(f"  Global features: {sample.global_features.shape}")
    print(f"  Target (y): {sample.y.shape}")
    
    print("\nNode features for first 3 cores:")
    print(sample.x[:3])
    
    print("\nGlobal features:")
    print(sample.global_features)
    
    print("\nTarget:")
    print(sample.y)