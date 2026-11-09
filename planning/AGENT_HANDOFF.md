# Execution handoff for the next analyst or AI agent

## Task and fixed inputs

Continue Part 2: (1) evaluate HRS-to-NLSY79 prediction of follow-up cognition; (2) test whether HRS-derived expected change improves an NLSY79 model. Read slides 5-7 of `project_introduction.pptx`, `NEXT_PHASE_GUIDE.md`, and `CODE_ADAPTATION.md` before running the full experiment.

Use branch `part2-data-preparation`. This branch root is the Part 2 project; there is no enclosing `part2/` directory. All source, documentation, identifiers, and output labels should remain English.

The collaborator already has two frozen archives. Do not replace or repackage them. Place copies under `local_data/`, or pass their existing absolute paths to the commands below.

| File | SHA-256 |
| --- | --- |
| `01_harmonized_cohort_data_v3.zip` | `639e5985a3f72ec1c982f1ac6208fb192bbae64c655141a73eae74dff06a6058` |
| `02_model_ready_cohort_pairs_v1.zip` | `c9d7c62bc284b56d49982e1ebba151b621501da6b2cb78e531c5720ec5d57f20` |

The repository's `metadata/` supplements these archives. Stage 02 is sufficient for the implemented primary training workflow. Stage 01 is needed for metadata rebuilding and new cohort exports. The Stage 01 archive already contains harmonized data; this repository's minimal processing entry starts there, not from original RAND/NLSY provider downloads.

## 1. Environment and validation

Use a dedicated Python 3.9-3.11 environment. The delivered checks used Python 3.9.7; package versions are pinned in `requirements.txt`. On Linux/UNC, activate an environment before running these commands. On Windows, invoke the environment's `python.exe` or activate its `Scripts` directory.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m part2_models validate --data local_data/02_model_ready_cohort_pairs_v1.zip
```

Expected data: HRS 52,023 rows, 8,299 people, 103 columns; NLSY79 2,305 rows/people, 185 columns. The loader checks the exact uncompressed table hashes. It does not mutate the ZIPs. The unified catalog covers 32 Stage 01 tables plus both Stage 02 tables, with 497 distinct field names overall. The modeling dictionary has 288 cohort-column entries and 186 distinct names.

Optional metadata rebuild:

```sh
python build_metadata.py local_data/01_harmonized_cohort_data_v3.zip local_data/02_model_ready_cohort_pairs_v1.zip
```

If a collaborator has only Stage 01, reconstruct Stage 02 to a NEW path:

```sh
python prepare_model_ready.py local_data/01_harmonized_cohort_data_v3.zip --output local_data/rebuilt_stage02.zip
```

The generated CSV hashes must match the delivered tables. ZIP container timestamps may differ; no existing archive is overwritten.

## 2. Engineering pilot

Run all three stages with `pilot_config.json`. It retains 15 deterministically selected households per available fold, including all their eligible pairs. The source pilot uses five folds; downstream tuning uses only development folds 0-3. Pilot predictions and models are engineering artifacts, not scientific results.

```sh
python -m part2_models source --data local_data/02_model_ready_cohort_pairs_v1.zip --config planning/pilot_config.json --out runs/pilot/source
python -m part2_models transfer --data local_data/02_model_ready_cohort_pairs_v1.zip --source runs/pilot/source --out runs/pilot/transfer
python -m part2_models target --data local_data/02_model_ready_cohort_pairs_v1.zip --config planning/pilot_config.json --transfer runs/pilot/transfer --out runs/pilot/target
```

Accept the pilot only when every stage has `_SUCCESS.json`, every saved model has a successful replay check, transferred predictors are finite for all 2,305 NLSY79 people, and the target prediction file has 460 rows with no outcome column. The evaluator rejects pilot runs. Do not use pilot model rankings to choose the full-run learner or search grid.

There are 13 selected pipeline artifacts: five source models (three endpoint learners, one change model, one simple reference) and eight target models (two learners times C/A/B_score/B_change). The persistence reference requires no fitted model.

## 3. Freeze the full experiment specification

Use `experiment_config.json`. Record the Git commit and `pip freeze` output in the run directory. The current grids require 80 HRS cross-validation fits, five HRS final fits, 128 NLSY79 development cross-validation fits, and eight NLSY79 final fits. Allocate runtime after measuring the pilot; larger full training samples can make extrapolation nonlinear. The implementation uses CPU and four threads by default.

The primary target learner is CatBoost; the primary increment is B_change. B_score preserves the earlier endpoint-feature proposal as a prespecified secondary comparison. This makes the change-model comparison in slide 6 explicit. Any change to this hierarchy, grid, cohort, or feature list must be recorded before locked-test evaluation.

Maintain these invariants:

- Age 48-55 at baseline, actual elapsed time 4-16 years, strict observed 27-point target.
- Shared 29 features for the primary source model; keep elapsed years and birth year.
- HRS pair weights are recomputed within each training subset, then normalized.
- Household folds are fixed; NLSY79 fold 4 never enters preprocessing, tuning, early stopping, or training.
- Source learner selection uses HRS development CV only. NLSY79 outcomes never select the source learner.
- Preprocessing is fitted separately inside every training fold. Candidate features are selected explicitly; outcome and follow-up assessment metadata do not become baseline inputs.
- Keep all predictions on their original regression scale; primary evaluation does not clip to 0-27 or recalibrate on the test set.
- RAND education's per-person measurement wave is unresolved in this export. Run the education-excluded sensitivity. Do not claim fully verified prospective timing for this field without additional wave-specific evidence.

## 4. Train and persist HRS source models

```sh
python -m part2_models source --data local_data/02_model_ready_cohort_pairs_v1.zip --config planning/experiment_config.json --out runs/main/source
```

The endpoint learner is selected by pooled, person-balanced out-of-fold RMSE across five household folds. Ties follow configuration order. These selection scores are development estimates. Each learner's selected configuration is refitted on all eligible HRS rows and saved. The conditional-change ridge penalty is selected separately. Read `source/selection.json` for the selected endpoint model; do not hard-code a learner based on pilot performance.

Each learner directory includes the complete search table, selected-candidate out-of-fold predictions, training-row/weight manifest, and saved pipeline. Repeated invocations reuse only an intact completed stage with the identical contract. Preserve failed/incomplete directories and start a new run identifier after diagnosing a failure.

## 5. Generate and freeze transferred features

```sh
python -m part2_models transfer --data local_data/02_model_ready_cohort_pairs_v1.zip --source runs/main/source --out runs/main/transfer
```

`transfer_predictions.csv.gz` has all 2,305 target participants and contains no observed target outcome. It includes S_score, S_delta, trajectory_endpoint, endpoint_delta, simple_prediction, and persistence_prediction. `prediction_manifest.json` records the source model hashes. Verify S_delta equals trajectory_endpoint minus baseline cognition. Preserve this file through downstream model comparisons.

## 6. Train NLSY79 comparisons and freeze test predictions

```sh
python -m part2_models target --data local_data/02_model_ready_cohort_pairs_v1.zip --config planning/experiment_config.json --transfer runs/main/transfer --out runs/main/target
```

Fit C (29), A (111), B_score (112), and B_change (112) before model-specific encoding. Each learner and variant selects its own parameters using the same four development folds and the same search grid. It then fits all 1,845 development participants and predicts the same 460 locked-test participants. Test outcomes are not used to fit, select, or score these models. Keep `locked_predictions.csv.gz` and all fitted artifacts immutable.

## 7. Evaluate the locked test once the planned predictions exist

This is the first step that computes locked-test performance. Finish the prespecified training and freeze all planned comparison predictions before invoking it. The evaluator checks the training configuration and transfer hashes and refuses pilot artifacts.

```sh
python -m part2_models evaluate --data local_data/02_model_ready_cohort_pairs_v1.zip --config planning/experiment_config.json --transfer runs/main/transfer --target runs/main/target --out runs/main/evaluation
```

Deliver `test_metrics.csv`, `paired_improvement.csv`, `scored_test_predictions.csv.gz`, and `evaluation.json`. Direct-transfer and downstream metrics use the identical held-out people. The main improvement is RMSE(A)-RMSE(B_change) for CatBoost with a paired household bootstrap interval. There are 410 test households; each bootstrap draws households with replacement and includes all members, using the same draw for both predictions. The 2,000-repeat interval quantifies fixed-model test uncertainty. It does not include training/model-selection uncertainty. Follow-up bands are descriptive stratifications, not independent cohorts.

Do not tune, select a different headline comparison, or re-split participants after seeing test results. Report negative or null improvement as observed. Do not fabricate performance or interpret low cognitive-score thresholds as validated diagnoses.

## 8. Reload models without retraining

Example for the separately fitted change model:

```sh
python -m part2_models predict --data local_data/02_model_ready_cohort_pairs_v1.zip --model runs/main/source/trajectory/model --cohort NLSY79 --out runs/replay_trajectory.csv
```

Example for the primary downstream model:

```sh
python -m part2_models predict --data local_data/02_model_ready_cohort_pairs_v1.zip --model runs/main/target/catboost/B_change/model --transfer runs/main/transfer --cohort NLSY79 --out runs/replay_target.csv
```

Replay the 460 locked participants against the frozen target prediction file by pair_id; maximum absolute difference must be at most 1e-10. The second replay includes development participants as well; those in-sample predictions are not evaluation results. Preserve the recorded environment to load the artifacts.

## 9. Additional analyses and publication outputs

For implemented sensitivities, copy the configuration and use a separate run directory: `source_feature_set=common_expanded77`, `first_pair_only=true`, or `exclude_education=true`. Complete their source/transfer/target stages before the corresponding evaluations. Keep the 2,305-person target sample fixed for these comparisons. Education exclusion removes four indicators from both cohorts' inputs.

Broader age ranges, three-component baseline cognition, official imputed cognition, and survey-weighted analyses require new explicit table contracts and exports from Stage 01. They are planned extensions, not enabled by changing a flag in the frozen-data loader. First quantify participant overlap, missingness, interval support, and source/target compatibility; validate the new tables and metadata; then add a versioned loader/configuration. Do not bypass the current checksum checks to force a new cohort into the old experiment.

Produce a sample-flow table, baseline/missingness table, direct-transfer table, A/B improvement table, and sensitivity table. Save the complete run artifacts and a concise findings report with exact data/code versions. Use actual stored predictions for figures. Keep performance summaries in numerical tables and honor the project's figure preferences.

## Copyable continuation instruction

> Work from this branch root. Read planning/README.md, planning/NEXT_PHASE_GUIDE.md, planning/CODE_ADAPTATION.md, and this handoff. Locate the two frozen archives without editing them. Validate the data and run tests. Run the engineering pilot, verify persisted pipelines and absence of test-outcome scoring, then execute the prespecified full source, transfer, and downstream training stages. Preserve household folds and immutable run manifests. Complete planned prediction files before final evaluation. Report the actual locked-test comparisons and saved artifact paths, with the education provenance limitation. Implement any additional cohort or trajectory extension as a separately versioned, validated analysis; never silently change the primary sample or use held-out outcomes for model selection.
