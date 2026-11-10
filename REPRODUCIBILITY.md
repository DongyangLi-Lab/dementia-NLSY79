# Completed revised analysis: code and execution map

## Entry points and inputs

The current pipeline is `model/` → `train/` → `inference/`. The root legacy notebooks, `results/` and `figures/` belong to BASE and are retained for historical reference. The current numerical results are in `delivery/tables/audited_sources/` and the local `artifacts/` tree.

Use Python 3.12 and `requirements-revised.txt`. The recorded local run used PyTorch 2.8.0 + CUDA 12.8. Install the appropriate PyTorch build for the target GPU; LightGBM additionally needs an OpenCL-capable build. The local TabICL v1.1 checkpoint (`tabicl-classifier-v1.1-0506.ckpt`) must be available in the Hugging Face cache. Raw data and model checkpoint licensing/access requirements remain those of the original sources.

Input tables go in `data/preprocessed data/`. Pass the authorized HRS 2022 Tracker path explicitly:

```powershell
python run_benchmark.py --tracker /path/to/trk2022tr_r.dta --prepare-only
python run_benchmark.py --tracker /path/to/trk2022tr_r.dta
python run_table_sources.py
python -m inference.death_audit --tracker /path/to/trk2022tr_r.dta
```

Use `python run_benchmark.py --help` to inspect the runner options. The recorded coordinator maps models to two GPU lanes; use the explicit model/device module interfaces on a differently configured machine. Only one process per GPU lane should execute at a time. Do not launch a second coordinator while one is active.

## Completed distributed run

`cluster/` contains imputation/model dispatch and verified receiving. `artifacts/*.py` and selected `artifacts/unc/` source files are deliberately tracked despite the output directory's default ignore rule: they contain the actual frozen recovery, inference, validation and finalization code used for this run. Diagnostic scripts retain their historical paths and must not be mistaken for the main entry point. Their source bytes are preserved for hash-based receipts.

Actual devices differed from the initial lane plan: LR/GNB/DT used CPU; CatBoost/RealMLP used RTX 3070 Ti; LightGBM/TabICL used RTX 5070 Ti; GAT primary 6-year used RTX 5070 Ti and its other three combinations used UNC L40S. The formal TabICL SHAP used RTX 5070 Ti for indices 0–24 and UNC L40S for 25–499.

The UNC production worker, submission script, and cuBLASLt C++ source are in `artifacts/unc/guarded_l40s_8756544/`. Its wrapper depends on the CUDA/cuBLASLt ABI from the recorded environment and frozen CPU-transformed inputs. The local private output archive preserves the exact compiled `.so`, inputs, controls and receipts. A newly compiled binary requires fresh controls; never change an expected hash merely to bypass a failed check. Frozen input paths identify this completed run, not a generic promise of portability.

For the already verified result tree, the finalization sequence is:

```powershell
python artifacts/register_guarded_tabicl_shap.py
python artifacts/finalize_tabicl_importance.py --cross-device
python artifacts/finish_verified_sources.py
python artifacts/audit_final_completion.py
```

These commands validate and write results. For release inspection without rerunning scientific stages, use `python tools/verify_delivery.py`. Original full-audit evidence is in `delivery/verification/`.

## Presentation and packaging

```powershell
python tools/build_outcome_flow_table.py
python tools/prepare_delivery.py
python tools/verify_delivery.py
python tools/package_delivery.py
python tools/package_outputs.py revised
python tools/package_outputs.py history
```

The S1 renderer needs `pdflatex`, `booktabs`, `tabularx`, `newtxtext` and `newtxmath`. Runtime tables are intentionally output-only, per the delivery request; no runtime rendering code or temporary material is included. The original runtime measurements and audited S20 remain reproducible through `inference/evaluation.py` and `inference/supplementary_tables.py`.

Historical document/figure builders and their existing inputs are preserved in the local historical archive, with their original relative layout. They may refer to local paths and older cohorts; they are not the current revised manuscript renderer.
