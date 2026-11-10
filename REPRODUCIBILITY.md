# Completed revised analysis: code and execution map

## Latest retention decision: outputs and models separated

The user requested permanent deletion of local UNC working directories, diagnostic tensors, transfer bundles, temporary replay/control material and the former mixed archives after completion. Final predictions, 500-by-60 TabICL SHAP, importance tables, cohorts, 352 model checkpoints and 44 preprocessing caches remain intact. Use `packages/result.zip` for outputs and `packages/model.zip` for model/preprocessing files. The saved completion reports describe the successful audit before cleanup; deleted diagnostics can no longer be replayed. No experiment was rerun for cleanup. The distributed execution description below is historical provenance, not a set of currently available scratch files.

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

The UNC production worker and submission script are in `artifacts/unc/guarded_l40s_8756544/`. The matching dynamic-row cuBLASLt source, recorded build command and source/ABI evidence are in its `production_build/` subfolder (`try_split_rows`, not the older fixed-row `try_split` diagnostic). They were retrieved read-only from the original cluster; no rebuild or new GPU execution was performed for this release. The wrapper depends on the recorded CUDA/cuBLASLt ABI and frozen CPU-transformed inputs. The local private output archive preserves the exact compiled `.so`, inputs, controls and receipts. A newly compiled binary requires fresh controls; never change an expected hash merely to bypass a failed check. Frozen input paths identify this completed run, not a generic promise of portability.

The completed run originally used the following finalization sequence. After cleanup these historical commands require deleted diagnostics and should not be rerun against the retained tree:

```powershell
python artifacts/register_guarded_tabicl_shap.py
python artifacts/finalize_tabicl_importance.py --cross-device
python artifacts/finish_verified_sources.py
python artifacts/audit_final_completion.py
```

These commands validate and write results. For release inspection without rerunning scientific stages, use `python tools/verify_delivery.py`; it also works from the extracted code/delivery ZIP without a Git checkout or private result archive. In that case it verifies the copied audited sources and saved evidence, rather than accessing participant-level records. Original full-audit evidence is in `delivery/verification/`.

## Presentation and packaging

```powershell
python tools/build_outcome_flow_table.py
python tools/prepare_delivery.py
python tools/verify_delivery.py
python tools/package_result_model.py result
python tools/package_result_model.py model
```

The S1 renderer needs `pdflatex`, `booktabs`, `tabularx`, `newtxtext` and `newtxmath`. Runtime tables are intentionally output-only, per the delivery request; no runtime rendering code or temporary material is included. The original runtime measurements and audited S20 remain reproducible through `inference/evaluation.py` and `inference/supplementary_tables.py`.

Selected historical presentation outputs are included in result.zip and labelled historical. Their original source directories outside revised/ remain in the project, but the former historical ZIP is removed. They are not the current revised manuscript renderer.
