#!/bin/bash
#SBATCH -n 1
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output ./logs/ic-%j.log

export PYTHONUNBUFFERED=TRUE

python src/ecmointerp/dataProcessing/inclusionCriteria.py