# Revised calculation outputs

## Post-completion retention update

The user requested permanent cleanup after final verification. UNC scratch files, transfer packages, calibration/replay tensors, temporary logs and the old mixed archives have been removed from the current working tree. Final cohorts, predictions, importance/SHAP, comparison results and tables remain intact. Checkpoints and preprocessing caches are retained separately in `packages/model.zip`; outputs are in `packages/result.zip`. Compact saved audit reports describe the completed pre-cleanup run. The historical distributed details below do not imply that deleted input/control files can still be replayed.

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

## Distributed TabICL importance and completion evidence

The formal KernelSHAP design remains the BASE configuration: 500 participants, 800 coalitions, 10 background participants, and 60 predictor groups. The local RTX 5070 Ti owns indices 0–24; the UNC L40S array owns 25–499. `shap_dispatch_ownership.json` records this allocation. Local workers checkpoint individual participants and use bounded fresh-process batches to release accumulated host memory.

UNC uses the exact locally exported CPU transformations. Cross-device predictions did not pass the original strict elementwise tolerance. The recorded decision in `tabicl_cross_device_acceptance.json` accepts production under per-job probability, SHAP, ranking, and direction checks; it is not a statistical equivalence claim. `tabicl_shap_run_inventory.json` lists independently validated downloaded runs. Each subject's SHAP calculation uses background and explained-cohort predictions from that same execution. Raw arrays, controls, execution routes, and source hashes remain available.

`tabicl_permutation_execution_audit.json` covers the replay of potentially contended local permutation groups. Original mismatching values and bootstrap files are preserved under `tabicl_execution_reconciliation/`; current results use the validated exclusive replay, with affected bootstrap summaries recomputed.

After all participants have arrived, the CPU finalization sequence is:

```powershell
.venv-cu128/Scripts/python.exe artifacts/register_guarded_tabicl_shap.py
.venv-cu128/Scripts/python.exe artifacts/finalize_tabicl_importance.py --cross-device
.venv-cu128/Scripts/python.exe artifacts/finish_verified_sources.py
.venv-cu128/Scripts/python.exe artifacts/audit_final_completion.py
```

The final source builder reuses all 160 verified mean-imputation/complete-case outputs, including their actual remote hardware provenance; it does not refit them on local hardware. It requires all eight importance receipts before writing the 23 numbered tables. `completion_audit.json` is written only after the complete training, verification, evaluation, importance, and table scope passes. Its absence means the complete computation has not yet been certified. `source_status.json` separately records final table hashes. Historical partial-table folders are excluded from the final source manifest. Figure rendering and manuscript document updates remain outside this computation pass.
