# HRS and NLSY79 data preparation

| Stage | Archive | Contents |
| --- | --- | --- |
| 01. Harmonized cohort data | `01_harmonized_cohort_data_v3.zip` | 32 processed tables: HRS person-wave records, NLSY79 baseline and life-course data, cohort alignment, analysis variants, and household folds. The archive also includes a variable dictionary and validation metadata. |
| 02. Model-ready cohort pairs | `02_model_ready_cohort_pairs_v1.zip` | Two analysis tables with the feature configuration and usage notes. HRS has 52,023 pairs from 8,299 people. NLSY79 has one pair for each of 2,305 people. |

## Build Stage 02

From this directory, place the Stage 01 archive under `local_data/` and run the following command with Python 3.9 or newer. The script uses only the standard library.

```sh
python prepare_model_ready.py local_data/01_harmonized_cohort_data_v3.zip
```

The script writes `local_data/02_model_ready_cohort_pairs_v1.zip`. It can also take the original `Part2_HRS_complete_v3_UNC_final.zip` as input. Use `--output PATH` to select a different output location. The build checks input hashes and verifies the generated CSV content against the validated model-ready export.

Stage 02 uses baseline ages 48–55, actual follow-up intervals of 4–16 years, and the strictly observed 27-point cognition outcome. `followup_years` is a model input. `followup_band` is for reporting. Use `features.json` to select predictor columns and `household_id` and `cv_fold5` to preserve the existing grouped splits. The NLSY79 table includes the 82 additional life-course candidate predictors.

The build begins with the harmonized data, combines the selected inputs, and packages the two tables. Cohort extraction, harmonization, and model fitting are separate stages.
