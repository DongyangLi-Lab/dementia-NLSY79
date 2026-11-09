# Revised calculation outputs

| Path | Contents |
| --- | --- |
| `cohorts/` | Four primary and four sensitivity cohorts, per-person label audit ledgers, input and output hashes |
| `imputation/` | One training-only cache for each repeat and temporal split; predictors and row membership |
| `checkpoints/` | Selected internal and temporal models, validation grids, thresholds, backend metadata, receipts |
| `predictions/` | Participant probabilities, labels, completeness flags, repeat-level and temporal metrics |
| `importance/` | Permutation probabilities, importance and uncertainty, SHAP and rank agreement |
| `imputation_comparison/` | Independent mean-imputed and complete-case held-out runs |
| `tables/` | Cohort flow and death checks, S1, baseline summaries, performance, CIs, calibration, decision curves, runtimes, importance, imputation comparison |
| `tables/supplementary/` | Single main table and submitted supplementary table data by number |
| `logs/` | CPU and GPU job logs, stage status and device probes |

Use `tables/participant_flow.csv` and `tables/S1_outcome_flow.csv` for Figures 1 and S1. S1 rows marked `subset` are included in their parents. Use the saved repeat metrics and temporal predictions for Figures 3–4 and S3–S7. Use `tables/imputation_comparison_repeats_h6.csv` for S2 and the `tables/importance_*` sources for Figure 5 and S8–S9. Check `source_status.json` before using the complete set. Supplemental predictor definitions S2–S8 are in the Part 1 predictor dictionary.
