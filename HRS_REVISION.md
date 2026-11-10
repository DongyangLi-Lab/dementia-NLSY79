# Revised HRS benchmark

The completed run and release supersede the initial code-only publication. See `README.md`, `REPRODUCIBILITY.md` and `delivery/README.md` for the final scope and actual hardware. The commands below are module-level examples. All 44 caches and `artifacts/prepared_inputs.json` must be validated before training. Set GPU visibility and LightGBM OpenCL device for the target host.

The revised code separates model definitions, training, and held-out inference. It updates cohort eligibility and outcome definitions for the HRS dementia analysis.

| Directory | Purpose |
| --- | --- |
| `model/` | Definitions and parameter grids for the eight model families |
| `train/` | Cohort construction, predictor-only imputation, validation tuning, and checkpoint creation |
| `inference/` | Held-out prediction, verification, and analysis of saved predictions |

Install a PyTorch build appropriate for your device, then install the dependencies with `python -m pip install -r requirements-revised.txt`. Run the commands from the repository root with the HRS inputs available. Supply the path to your HRS Tracker file.

```sh
python -m train.cohorts --tracker /path/to/tracker.dta
python -m train.imputation --horizon 6 --outcome final
python -m train.imputation --horizon 8 --outcome final
python -m train.imputation --horizon 6 --outcome any
python -m train.imputation --horizon 8 --outcome any
python -m inference.verify --prepared-only
python -m train.benchmark --model LR --horizon 6 --outcome final --device cpu
python -m inference.predict --model LR --horizon 6 --outcome final
```

The examples fit and evaluate one model on one machine. Repeat the last two commands for the desired model, horizon, and outcome. For GPU models, select one visible device and pass `--device 0` to training. LightGBM requires an OpenCL-enabled build. TabICL uses the original v1.1 checkpoint.

The pipeline fits imputation on the training subset of each internal split and applies it to that split's validation and test sets. The temporal analysis fits imputation on the full development cohort and applies it to the temporal cohort. All models reuse those prepared inputs. The revised entry points write generated files beneath `artifacts/`. The original benchmark files remain available for reference.
