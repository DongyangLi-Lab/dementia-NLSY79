# Revised HRS dementia benchmark

This branch contains the completed revised computation based on `BASE` commit `9d34fe46270862a7b867ab3b919bb6971714cd49`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for inputs, environments, entry points and the distributed execution map.

**Latest delivery:** [formatted tables and structured results](delivery/README.md). Local output archives are now separated into `packages/result.zip` and `packages/model.zip`; the former mixed archives and UNC working/diagnostic materials were removed at the user's request. The runtime table and English-status S1 use matching black three-rule typography. The earlier English-status S1 is preserved as a historical reference; the current S1 uses final audited counts.

**Result-only plotting:** `python run_figures.py --result packages/result.zip --output outputs/result_figures` rebuilds the current five main figures, three supporting figures and all 23 numbered tables without checkpoints or training. `make_figures.py` is now an alias for this entry point. See [FIGURES.md](FIGURES.md) for input validation, stage selection, the historical figures that require additional sources, and the exact completion boundary.

| Directory | Responsibility |
| --- | --- |
| `model/` | Eight model definitions, tuning grids, and device settings |
| `train/` | Cohort assembly, predictor-only imputation, validation tuning, and checkpoint writing |
| `inference/` | Checkpoint replay, verification, metrics, importance, and tabular sources |
| `cluster/` | Optional recorded local/UNC dispatch and verified receiving |
| `artifacts/` | Local calculation outputs and frozen execution/verification scripts |
| `tools/` | English-status S1 rendering, delivery verification and archive packaging |
| `delivery/` | Final aggregate tables, runtime estimates, completion evidence and labelled historical presentations |
| `packages/` | Local archives, ignored by Git; includes participant-level outputs |

## Scientific scope

The four windows are 2000–2006, 2000–2008, 2010–2016 and 2010–2018. Primary outcomes require confirmation of survival at the final wave. Death flags and death dates override earlier cognitive observations. An incident case must have observed dementia during follow-up and survive to the final wave. A non-case must be dementia-free at the final assessment. Unresolved outcomes, loss to follow-up and deaths are excluded. The sensitivity analysis retains death exclusion but allows any dementia-free follow-up assessment for confirmed survivors with no observed dementia.

The 60 predictor groups yield 94 model inputs. ID, baseline dementia status and outcome Y are excluded from imputation. Each of ten stratified 60/20/20 development splits fits miceforest once on training predictors and applies that fitted imputer to validation and test. Internal final models refit on the already-imputed train and validation data; no second train-plus-validation imputer is fitted. Temporal imputation and model fitting use the full 2000 cohort; the 2010 cohort is reserved for evaluation.

The completed run includes **44 imputation caches, 32 model/outcome/horizon combinations, 352 saved fitted checkpoints, 160 missing-data comparison outputs, eight-model importance and 23 final numbered tables**. Formal TabICL KernelSHAP uses 500 participants, 800 coalitions and 10 background participants over 60 predictor groups. See [completion evidence](delivery/verification/completion_audit.json).

Actual hardware is recorded per run. GAT ran on both RTX 5070 Ti and UNC L40S across different combinations; CatBoost and RealMLP used RTX 3070 Ti. The runtime workbook preserves measured time and marks normalized columns as idealized estimates, not same-device measurements. The original strict cross-device TabICL check remains failed; final distributed SHAP passed separately recorded engineering acceptance checks. This is not a claim of strict numerical equivalence.

## Reproduce or inspect

Install the recorded dependencies and a compatible PyTorch/CUDA build. Obtain authorized HRS inputs and the TabICL v1.1 checkpoint separately. From the repository root:

```powershell
python run_benchmark.py --tracker /path/to/trk2022tr_r.dta --prepare-only
python run_benchmark.py --tracker /path/to/trk2022tr_r.dta
python run_table_sources.py
```

The coordinator reproduces the recorded two-GPU lane plan. On another hardware layout, use the explicit module interfaces described in [HRS_REVISION.md](HRS_REVISION.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Check `--help` before starting. Do not duplicate an active coordinator or bypass receipt failures. Completed checkpoints are reused only when their source receipts match.

For the saved delivery, no training is needed:

```powershell
python tools/build_outcome_flow_table.py
python tools/prepare_delivery.py
python tools/verify_delivery.py
```

Runtime rendering scripts and temporary assets are intentionally omitted at the user's request. Runtime outputs, their assumptions and original measured timing records are retained.

## Historical outputs and data access

The tracked BASE `results/`, `figures/`, notebooks and predictor dictionary retain historical provenance. The historical dictionary's missing-count/complete-case description is not the revised imputation protocol; use the description above and current predictor tables. Selected earlier Word/figure outputs are separated in `delivery/historical_reference/`; they have not been regenerated from the completed revised computation. In particular, the old 128-person SHAP figure is not the formal 500-person result. Manuscript, supplement, reviewer-response and full figure revisions remain a separate publication task.

Raw HRS inputs, participant-level predictions, imputation caches and model checkpoints are not added to this public release. Predictions are in local `result.zip`; checkpoints and preprocessing caches are in `model.zip`. Saved verification reports describe the completed run before cleanup; deleting diagnostic tensors means the original full cross-device replay cannot be repeated from the retained files alone. Obtain original data through [HRS](https://hrs.isr.umich.edu/).
