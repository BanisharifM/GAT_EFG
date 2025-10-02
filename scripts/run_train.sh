#!/bin/bash
#SBATCH --job-name=gat_train
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/train_%j.out
#SBATCH --error=experiments/logs/train_%j.err

# Create log directory
mkdir -p experiments/logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Change to project directory
cd /work/hdd/bdau/mbanisharifdehkordi/GAT_EFG

# Use direct python path from your conda env
PYTHON_PATH="$HOME/.conda/envs/gnn4_env/bin/python"

# Print info
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_PATH"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU info not available')"
echo "=================================================="

# Run training
srun $PYTHON_PATH scripts/train.py \
    --config configs/model_config.yaml \
    --experiment exp_001_gat_baseline

echo "=================================================="
echo "Training completed at: $(date)"
echo "=================================================="
