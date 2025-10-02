"""Test dataset creation and data loading."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import CoreGraphDataset, create_dataloaders


def test_dataset():
    print("=" * 60)
    print("Testing Dataset Creation")
    print("=" * 60)
    
    dataset = CoreGraphDataset(
        csv_path='data/splits/train.csv',
        metadata_path='data/processed/merged_metadata.json'
    )
    
    sample = dataset[0]
    
    assert sample.x.shape == (8, 3), f"Wrong node feature shape: {sample.x.shape}"
    assert sample.edge_index.shape[0] == 2, "Wrong edge_index format"
    assert sample.global_features.shape == (1, 8), f"Wrong global features: {sample.global_features.shape}"
    assert sample.y.shape == (1,), f"Wrong target shape: {sample.y.shape}"
    
    print("\n✅ Dataset structure validated")
    
    print("\n" + "=" * 60)
    print("Testing DataLoaders")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, feature_dims = create_dataloaders(
        train_path='data/splits/train.csv',
        val_path='data/splits/val.csv',
        test_path='data/splits/test.csv',
        batch_size=32,
        num_workers=0
    )
    
    batch = next(iter(train_loader))
    
    print(f"\nBatch structure:")
    print(f"  x: {batch.x.shape}")
    print(f"  edge_index: {batch.edge_index.shape}")
    print(f"  global_features: {batch.global_features.shape}")
    print(f"  y: {batch.y.shape}")
    print(f"  batch: {batch.batch.shape}")
    
    expected_global_shape = (32, 8)
    assert batch.global_features.shape == expected_global_shape, \
        f"Wrong global_features shape: {batch.global_features.shape}, expected {expected_global_shape}"
    
    print(f"\n  ✅ Global features correctly batched: {batch.global_features.shape}")
    
    print(f"\nFeature dimensions:")
    print(f"  Node feature dim: {feature_dims[0]}")
    print(f"  Global feature dim: {feature_dims[1]}")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_dataset()