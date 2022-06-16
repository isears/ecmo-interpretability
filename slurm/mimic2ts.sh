#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output ./logs/mimic2ts-%j.log


cd ~/Repos/tabnet-sepsis/

python3 -m mimic2ts --timestep 21600 --blocksize 25000000 mimiciv mimicts 
