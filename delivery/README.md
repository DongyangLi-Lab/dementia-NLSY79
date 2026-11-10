# HRS revised results delivery — 10 November 2026

## Current results

| Folder | Contents |
| --- | --- |
| `tables/audited_sources/` | Final aggregate CSV sources, including one manuscript Table 1 and 22 supplementary tables; original audited numbers retained |
| `tables/outcome_flow/` | Current S1 in PDF, PNG, LaTeX and CSV; English outcome statuses and final revised counts |
| `tables/runtime/` | Three-rule runtime table in PDF/PNG, editable Excel workbook, and summary CSV |
| `verification/` | Saved completion, source, model verification, comparison and cross-device acceptance evidence |
| `historical_reference/` | Earlier English-status S1, selected figure suite, captions and Word documents; historical numbers, not regenerated final results |

The computation is complete: 44 imputation caches, 32 model/outcome/horizon combinations, 352 fitted checkpoints, 160 missing-data comparison outputs, eight model importance outputs and the formal 500-participant TabICL KernelSHAP analysis. Full manuscript/figure regeneration is not claimed by this delivery.

### English-status S1

The earlier table used `Positive`, `Negative`, `Excluded`, `Baseline` and `Positive or negative`, without parenthesized codes. That exact PDF and LaTeX are preserved in `historical_reference/english_status_S1/`. The current table in `tables/outcome_flow/` uses the same style and statuses but reads the final audited S1 counts. Its four analytic cohort sizes are **13,246; 12,236; 14,123; 11,603**. Indented rows remain subsets and are not added twice. The current renderer also checks cohort accounting and death subcategory totals.

### Runtime interpretation

The workbook retains all 32 measured run summaries and editable hardware assumptions. Summary rows sum both outcomes and both horizons, including tuning and refits. The recorded fitting/prediction total is **521.4864 minutes**, not elapsed project time. Imputation, SHAP, permutation, bootstrap, failed attempts and queueing are excluded.

RTX 5070 Ti is the reference for two separate idealized estimates: source/reference FP32 capacity and source/reference memory bandwidth. CPU times remain unchanged. These estimates are not calibrated same-device measurements, confidence bounds or evidence of a fair hardware-controlled ranking. Hardware sources are included in the workbook. At the user's request, runtime rendering scripts, temporary LaTeX, intermediate JSON and preview/debug materials are not delivered or published. Original benchmark timing collection remains part of the analysis code.

### Source and history boundaries

`MANIFEST.json` records SHA-256 values for delivered files. Numerical source files are copied without changing the audited originals. The saved strict cross-device TabICL check remains failed; accepted production used the recorded per-job engineering controls. Original results, device metadata and reconciliation evidence are retained.

Earlier selected figures/Word documents still contain historical analyses, including the 128-person SHAP preview; they must not be presented as figures regenerated from the final 500-person result. Historical archives also contain rejected designs, which remain historical only. The current user choices and CI restrictions are recorded in the historical indexes.

## Local archive structure

The ignored `packages/` folder contains:

- `HRS_code_and_delivery_20261110.zip`: necessary code, environment requirements, final aggregate tables, formatted runtime/S1, selected historical presentation references and file checksums.
- `HRS_revised_private_outputs_20261110.zip`: full saved revised calculation outputs, preserving the `artifacts/` hierarchy, including checkpoints, imputation, predictions, SHAP and execution evidence. This archive contains HRS-derived individual data and is not uploaded to the public repository.
- `HRS_historical_outputs_20261110.zip`: complete earlier `output/` archive under `historical_outputs_NOT_CURRENT/`, retaining source code, intermediate artifacts and superseded designs.

Each archive has a per-file SHA-256 manifest and a CRC-tested verification sidecar. Raw HRS input files and Python environments are not outputs and are not redistributed. Obtain inputs through the project's existing HRS access process.
