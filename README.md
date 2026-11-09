# Revised HRS dementia benchmark

This worktree is based on branch `BASE` at `9d34fe46270862a7b867ab3b919bb6971714cd49`. The revised pipeline reads the baseline predictor exports and per-wave cognitive classifications in `data/preprocessed data/`, together with the local HRS 2022 Tracker. It derives labels again for 2000每2006, 2000每2008, 2010每2016, and 2010每2018.

| Directory | Responsibility |
| --- | --- |
| `model/` | Eight Part 1 model definitions, tuning grids, and device settings |
| `train/` | Cohort assembly, predictor-only imputation, validation tuning, and checkpoint writing |
| `inference/` | Checkpoint replay on held-out data, verification, metrics, importance, and tabular sources |
| `artifacts/` | New cohort ledgers, imputation caches, checkpoints, predictions, logs, and figure/table data |

The primary outcome requires Tracker confirmation of survival at the final wave. Death codes 5/6 in any follow-up wave and documented death years earlier than the final wave override every cognitive observation. A documented incident case is positive if the participant is alive at the final wave. A non-case must have a dementia-free final assessment. The sensitivity analysis allows any dementia-free follow-up assessment for confirmed survivors without an observed case. `inference.death_audit` independently compares retained labels with Tracker death flags and death/interview months. When death and a completed final interview share a month, the month alone cannot establish ordering and the Tracker vital classification applies.

A stratified 60/20/20 split is made ten times per horizon and outcome. Each split fits one miceforest kernel on the 94 model inputs from training participants, with 20 iterations. The same kernel imputes validation and test. The final internal model is refitted on the concatenated imputed train and validation matrices. Temporal imputation and model fitting use the full 2000 development cohort. Validation AUPRC selects configurations and validation F1 selects thresholds. The 60 predictor groups are checked against the 94-column schema. Baseline dementia status, ID, and Y are outside the imputation input. The split is stratified on Y before imputation.

GPU 0 (RTX 5070 Ti) runs LightGBM via OpenCL and TabICL via CUDA. GPU 1 (RTX 3070 Ti) runs CatBoost, RealMLP, and GAT via CUDA. LR, GNB, DT, and miceforest use CPU. The four CPU preprocessing jobs run in parallel. Once all 44 caches pass cohort, split, predictor and hash checks, the CPU and GPU model workers start and read the same verified inputs. TabICL retains the Part 1 v1.1 checkpoint, ensemble grid, and automatic memory sizing. The GAT reuses the fixed reference graph's similarity block across query graphs.

## Reproduce or resume

Use the local `.venv-cu128` (Python 3.12, PyTorch 2.8.0 + CUDA 12.8, reused from PET_MIL). The local TabICL v1.1 `tabicl-classifier-v1.1-0506.ckpt` must be in the Hugging Face cache. On another machine, install a PyTorch build compatible with both GPUs and the packages listed in `requirements-revised.txt`.

```powershell
$python = ".venv-cu128/Scripts/python.exe"
& $python run_benchmark.py --tracker "E:/yjx/Part2/data/interim/hrs_official_2022/trk2022tr_r.dta"
& $python run_table_sources.py
& $python -m inference.death_audit
```

Use `run_benchmark.py --prepare-only` to run and verify preprocessing as a separate stage. The benchmark reuses checkpoints and caches only when their receipts match. Training workers require the shared `artifacts/prepared_inputs.json` receipt and reject changed cache files. If an existing run is active, inspect `artifacts/run_status.json`, `artifacts/source_status.json`, and the per-device logs. Start a resume only after the previous coordinator has exited. `run_table_sources.py` calculates the separate mean-imputation and complete-case retraining comparison for submitted Figure S2 and Table S12. The comparison uses the same split indices and restricts each subset to complete predictors for complete-case training and evaluation. The final source manifest is written after all required source files exist.

The tracked BASE results, figures and notebooks are historical. Use `artifacts/` for revised results. The submitted figure renderers and manuscript documents are outside this computation pass.
