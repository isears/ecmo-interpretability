#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output ./logs/singleFit-%j.log

module load cuda/11.3.1
module load cudnn/8.2.0
nvidia-smi

export PYTHONUNBUFFERED=TRUE


python --version

python src/ecmointerp/modeling/transformerSingle.py