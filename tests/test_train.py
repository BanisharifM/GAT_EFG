# Train BASELINE (no CFG) - for paper comparison
python scripts/train.py \
    --config configs/train_config_baseline.yaml \
    --experiment exp_baseline_no_cfg

# Train WITH CFG features - new approach
python scripts/train.py \
    --config configs/train_config_with_cfg.yaml \
    --experiment exp_005_gat_with_cfg