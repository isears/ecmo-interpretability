#!/bin/bash
#SBATCH -n 1
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output ./logs/featureSelection-%j.log

export PYTHONUNBUFFERED=TRUE

python src/ecmointerp/dataProcessing/featureSelection.py