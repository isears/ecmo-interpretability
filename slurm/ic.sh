#!/bin/bash
#SBATCH -n 1
#SBATCH -p debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output ./logs/ic-%j.log

export PYTHONUNBUFFERED=TRUE

python3 src/tabsep/dataProcessing/inclusionCriteria.py