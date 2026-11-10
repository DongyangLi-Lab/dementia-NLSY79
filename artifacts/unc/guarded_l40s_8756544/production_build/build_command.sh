#!/bin/bash
set -euo pipefail
module load cuda/12.9
g++ -std=c++17 -O2 -fPIC -shared -I.venv-model/lib/python3.12/site-packages/nvidia/cublas/include -I"$CUDA_HOME/include" icl_cublaslt_split_rows.cc -L.venv-model/lib/python3.12/site-packages/nvidia/cublas/lib -l:libcublasLt.so.12 -o icl_cublaslt_split_rows.so
