#!/bin/bash
#SBATCH --job-name=gat_cfg_log
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/train_cfg_log_%j.out
#SBATCH --error=experiments/logs/train_cfg_log_%j.err

cd /work/hdd/bdau/mbanisharifdehkordi/GAT_EFG
PYTHON_PATH="$HOME/.conda/envs/gnn4_env/bin/python"

srun $PYTHON_PATH scripts/train.py \
    --config configs/model_config_with_cfg_log.yaml \
    --experiment exp_007_gat_with_cfg_log
