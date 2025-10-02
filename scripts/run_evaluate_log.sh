#!/bin/bash
#SBATCH --job-name=eval_log
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=experiments/logs/eval_log_%j.out
#SBATCH --error=experiments/logs/eval_log_%j.err

cd /work/hdd/bdau/mbanisharifdehkordi/GAT_EFG
PYTHON_PATH="$HOME/.conda/envs/gnn4_env/bin/python"

srun $PYTHON_PATH scripts/evaluate.py \
    --experiment exp_002_log_transform \
    --checkpoint best.pth