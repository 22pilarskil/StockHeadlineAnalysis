#!/bin/bash
#SBATCH --job-name=distributed_training
#SBATCH --output=distributed_training_%j.out
#SBATCH --error=distributed_training_%j.err
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

# Load necessary modules (adjust based on your cluster's setup)
module load anaconda3/2023.03

# Activate your Python environment (if using conda or virtualenv)
conda activate DistributedTrain

cd "/home/hice1/lpilarski3/StockHeadlineAnalysis" && pwd && python3 -u train_distributed.py --distributed

echo "Training completed."