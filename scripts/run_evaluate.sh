#!/bin/bash
#SBATCH --job-name=gat_eval
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=experiments/logs/eval_%j.out
#SBATCH --error=experiments/logs/eval_%j.err

# Create log directory
mkdir -p experiments/logs

# Change to project directory
cd /work/hdd/bdau/mbanisharifdehkordi/GAT_EFG

# Use direct python path
PYTHON_PATH="$HOME/.conda/envs/gnn4_env/bin/python"

# Print info
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "=================================================="

# Run evaluation
srun $PYTHON_PATH scripts/evaluate.py \
    --experiment exp_001_gat_baseline \
    --checkpoint best.pth

echo "=================================================="
echo "Evaluation completed at: $(date)"
echo "=================================================="
