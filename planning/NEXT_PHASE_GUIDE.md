# Next-phase analysis guide

## Research questions

The project introduction asks whether an HRS-trained model can predict later NLSY79 cognition and whether HRS-derived information improves an NLSY79 prediction model. Complete the direct-transfer experiment first, then compare downstream models with and without the HRS-derived feature.

Slide 5 is implemented by endpoint regression with elapsed time as an input. Slide 6 is implemented by a separate conditional cognitive-change trajectory fitted to repeated HRS pairs. The exact trajectory equation and changes from the previous endpoint-only draft are in `CODE_ADAPTATION.md`.

## Current data contract

| Item | Specification |
| --- | --- |
| HRS source sample | 8,299 people, 6,543 households, 52,023 eligible pairs |
| NLSY79 target sample | 2,305 people, 2,028 households, one baseline-to-2020 pair per person |
| Baseline age | 48-55 completed years |
| HRS birth years | 1948-1964 |
| HRS baseline waves | Available eligible waves from 2004 onward |
| Follow-up | Actual interview interval of 4-16 years |
| Target | Strictly observed 27-point cognitive score at follow-up |
| Shared inputs | Primary 29, expanded sensitivity 77 |
| NLSY79 development sample | 1,845 people in 1,618 households |
| NLSY79 locked test sample | Fold 4, 460 people in 410 households |

Use one pooled HRS model with `followup_years` as a continuous input. Report performance separately for [4,8), [8,12), and [12,16] years using `followup_band`. NLSY79 first complete cognition was administered by birth cohort, followed by the 2020 repeat. Its current analysis table therefore has one pair per participant.

The primary shared features have substantially better NLSY79 availability than the expanded 77-field set. Select columns from the configuration, then apply model-specific handling inside training folds. Preserve the explicit source-code categories for occupation, retirement, household background, and other coded fields.

## Data preparation completion and pretraining checks

The current preparation release includes both cohort stages, unified metadata, reproducible Stage 01-to-02 code, and validation of the current model tables. Predictor missingness is retained for training-fold preprocessing.

Before fitting, verify archive and table checksums, person/pair uniqueness, exact feature order, finite score and duration ranges, household isolation, and the development/test assignments. Store package versions and random seeds. Enforce the metadata predictor lists in the data loader.

RAND education is a static cross-wave construction whose source wave is unavailable per person in this export. Document this source convention, predefine an education-excluded sensitivity, and obtain wave-specific evidence if the analysis requires fully verified prospective availability. NLSY79 predictor source-year checks in this release confirm that the recorded nonmissing source bounds are at or before each person's baseline.

Compute HRS pair weights as 1 divided by the person's pair count within each actual training subset. Normalize those weights within the subset. Preserve all household members within one fold. Fit imputation, categorical encoders, scaling, variable selection, and early stopping using the corresponding training data.

## Experiment 1 - direct transfer

Train ElasticNet, CatBoost, and LightGBM on the same HRS folds and primary shared inputs. Include a baseline-cognition persistence reference and a simple regression using baseline cognition, baseline age, and elapsed time.

Choose the source learner using person-balanced validation RMSE from HRS household-grouped five-fold cross-validation. Treat these scores as model-development estimates. Freeze the selected hyperparameters, preprocessing, and feature order, then refit on eligible HRS data.

Generate `S_score` for NLSY79 from its baseline inputs and actual time interval. Record the HRS model version and source-data checksum for each prediction output. The NLSY79 outcome is used at the final evaluation stage.

Report RMSE, MAE, R-squared, mean prediction error, and calibration intercept and slope. Report the frozen model's direct-transfer performance on the same locked NLSY79 test participants used in Experiment 2. Additional full-target external evaluation can be reported once all planned models and predictions have been frozen.

## Experiment 2 - incremental predictive value

| Model | Inputs |
| --- | --- |
| C | 29 common primary features |
| A | Common primary features plus 82 NLSY79-only candidate features, 111 before further encoding |
| B_score | A plus the HRS endpoint prediction S_score |
| B_change | A plus the HRS trajectory prediction S_delta; primary incremental comparison |

Use CatBoost regression for the primary A/B_change comparison, with ElasticNet as a secondary learner. Keep the same development households, target, and hyperparameter search budget for all variants. Tune within folds 0-3 of the development sample using household-grouped validation. Preserve the 460-person test set until the planned predictions and analysis choices are fixed.

The primary comparison is RMSE(A) minus RMSE(B_change). Positive values favor adding the HRS expected-change feature. Report the paired difference and a 95% interval from 2,000 shared household bootstrap samples, along with MAE, R-squared, and calibration. Report A versus B_score as the prespecified secondary comparison carried forward from the earlier plan. Bootstrap intervals describe uncertainty in the fixed-model test evaluation.

`S_delta` is the prediction from the separately fitted conditional-change trajectory. `endpoint_delta = S_score - baseline_cognition27` is exported as a distinct representation of the endpoint learner. With baseline cognition already present, S_score and endpoint_delta carry equivalent information. S_delta can differ because it comes from a different model and time structure. These definitions are fixed before test evaluation.

## Prespecified additional analyses

1. Fit the expanded 77-input source model and report its missingness burden.
2. Use one earliest eligible HRS pair per person to assess repeated-pair sensitivity.
3. Exclude the four education indicators to assess the static-source convention.
4. Use the three observed baseline components to recover participants affected by unavailable delayed recall. Evaluate the changed baseline information in the common participant subset as well as the expanded sample.
5. Compare baseline-age windows 48-55, 48-60, and 48-65 for HRS while holding the NLSY79 target set fixed. These broader source tables require separate exports from the harmonized person-wave data.
6. Evaluate official provider-imputed cognition and survey weighting as separately labeled sensitivities.
7. After freezing the main comparison, examine development-sample size and the contribution of AFQT, childhood background, and life-course histories using prespecified subsets.

The 3,199-person strict and 4,476-person three-component NLSY79 variants change age and follow-up support. Identify their scope explicitly when using them as additional analyses. Preserve the agreed primary sample when assessing the effect of a change in HRS training inputs.

## Outputs and execution order

Run the small household pilot to measure runtime and memory, then select local or UNC resources. The pilot exercises every learner across the existing folds using a small number of households. Complete source-model selection, freeze S predictions, train the downstream comparisons, and reveal the locked-test results once the prespecified prediction files are ready. Follow the exact commands in `AGENT_HANDOFF.md`.

Save fitted preprocessing and models, configuration and environment versions, fold assignments, training weights, validation search results, prediction files, and evaluation tables. Predictions should carry person ID, household ID, pair ID when applicable, cohort, model version, and actual follow-up duration. Keep row-level outputs in the project's data location.

Prepare a sample-flow table, a baseline/missingness table, a direct-transfer performance table, an A/B improvement table, and a sensitivity table. Use the project figure preferences when plotting observed results. Training and S generation are the next phase following this preparation release.
