#!/bin/bash
#SBATCH --job-name=hrs-model-setup
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-setup-%j.log
set -euo pipefail
umask 077
module load python/3.12.4
export PIP_CACHE_DIR="$PWD/.pip-cache" TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
python -m venv .venv-model
. .venv-model/bin/activate
python -m pip install --disable-pip-version-check 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --disable-pip-version-check 'numpy==1.26.4' 'pandas==2.2.3' 'scipy==1.13.1' 'scikit-learn==1.6.1' 'lightgbm==4.5.0' 'miceforest==6.0.3' 'pyarrow==21.0.0' 'catboost==1.2.8' 'imbalanced-learn==0.12.3' 'torch-geometric==2.6.1' 'pytabkit==1.6.1' 'tabicl==0.1.3' 'transformers==4.41.2' 'pytorch-lightning==2.2.1' 'torchmetrics==1.3.2'
python -m pip check
python -m cluster.model_dispatch verify-inputs
python -c 'import torch; from model.specs import model_specs; print("torch",torch.__version__,"cuda",torch.version.cuda,"models",list(model_specs()))'
