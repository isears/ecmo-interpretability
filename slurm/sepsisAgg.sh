#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output ./logs/sepsisAgg-%j.log

export PYTHONUNBUFFERED=TRUE

python3 src/tabsep/dataProcessing/sepsisAggregate.py