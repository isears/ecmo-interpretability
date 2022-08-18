#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output ./logs/attribute-%j.log

nvidia-smi

export PYTHONUNBUFFERED=TRUE


python --version

python src/ecmointerp/modeling/tstCaptum.py