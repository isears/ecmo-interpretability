#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output ./logs/unifiedPostProcessing-%j.log

export PYTHONUNBUFFERED=TRUE

echo "[*] Inclusion criteria..."
python3 src/tabsep/dataProcessing/inclusionCriteria.py
echo "[*] Precomputing meta..."
python3 src/tabsep/dataProcessing/precomputeMeta.py
echo "[*] Sepsis aggregation..."
python3 src/tabsep/dataProcessing/sepsisAggregate.py
echo "[*] Post processing..."
python3 src/tabsep/dataProcessing/postProcessTst.py

echo "[+] Done"