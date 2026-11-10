#!/bin/bash
#SBATCH --job-name=hrs-comparison
#SBATCH --partition=general
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-comparison-%A_%a.log
set -euo pipefail
umask 077
module load python/3.12.4
. .venv-model/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
models=(LR GNB DT)
model=${models[$SLURM_ARRAY_TASK_ID]}
/usr/bin/time -v -o "profile-comparison-${model}-${SLURM_JOB_ID}.txt" python -u -m cluster.model_dispatch run comparison final 6 "$model" "result-comparison-final-h6-${model}.zip"
sha256sum "result-comparison-final-h6-${model}.zip"
