#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output ./logs/cvLr-%j.log

export PYTHONUNBUFFERED=TRUE

python3 src/tabsep/modeling/timeseriesCV.py LR