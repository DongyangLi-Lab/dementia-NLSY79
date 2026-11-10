#!/bin/bash
#SBATCH --job-name=hrs_mice_36
#SBATCH --partition=general
#SBATCH --cpus-per-task=36
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
module load python/3.12.4
python -m venv .venv-impute
. .venv-impute/bin/activate
python -m pip install --disable-pip-version-check -r requirements-impute.txt
export HRS_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m cluster.unc_imputation run unc_plan.json unc_imputed.zip
