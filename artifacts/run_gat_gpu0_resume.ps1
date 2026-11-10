$ErrorActionPreference = 'Stop'
Set-Location 'E:\yjx\revised'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = 'GPU-c71ad67d-ba1b-b47e-f12d-491431d030eb'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '2'
$env:HF_HUB_OFFLINE = '1'
$env:HRS_OPENCL_DEVICE = '0'
$python = 'E:\yjx\revised\.venv-cu128\Scripts\python.exe'
& $python -u -m train.benchmark --horizon 6 --outcome final --model GAT --device 0 --threads 2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -u -m inference.predict --horizon 6 --outcome final --model GAT
exit $LASTEXITCODE
