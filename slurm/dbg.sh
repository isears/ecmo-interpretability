#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output ./logs/debug-%j.log


export PYTHONUNBUFFERED=TRUE
source /gpfs/home/isears1/anaconda3/bin/activate /users/isears1/anaconda/lsepsis

echo "Establishing connection back to $SLURM_SUBMIT_HOST:51277"
python3 -m debugpy --connect $SLURM_SUBMIT_HOST:51277 --wait-for-client $1