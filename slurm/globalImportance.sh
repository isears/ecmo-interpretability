#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
# #SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output ./logs/globalImportance-%j.log


export PYTHONUNBUFFERED=TRUE

python src/tabsep/reporting/globalImportance.py