# Survival eligibility for Part 2

Part 2 retains its continuous 27-point cognition outcome, baseline ages 48-55,
and actual follow-up intervals of 4-16 years. The Part 1 mortality correction
is applied to each HRS baseline-to-outcome pair, not as a binary dementia label
or a requirement that everyone survive until 2022.

Require official HRS Tracker ALIVE=1 at the pair's outcome wave. ALIVE=2 means
presumed alive and is insufficient. Any ALIVE=5/6 from baseline through the
outcome wave excludes the pair, even if a cognitive score was observed.
A known death year before the endpoint year, or a known death before the
actual interview, also overrides the score and an inconsistent alive flag.
For death and interview in the same year, an earlier known death month is
sufficient; equal months do not establish day-level ordering. A later death
does not remove an otherwise eligible earlier pair. Missing Tracker matches
and unconfirmed endpoint survival are excluded. The same rule applies to the
strict27 primary, baseline3, and official27 sensitivity definitions.

The local official source is `trk2022tr_r.dta`; its codebook is
`trk2022TR_R.txt`. Wave prefixes for 2000 through 2022 are GHJKLMNOPQRS.
ALIVE codes refer to wave vital status, not survival through December 31.

## Verified frozen data

The 2026-11-16 audit covers all six HRS branches in both all-pair and first-pair
form (12 tables), plus the model-ready HRS export. All rows pass. The primary
remains 52,023 pairs from 8,299 participants; official27 sensitivity remains
58,676 pairs from 8,608 participants. Consequently both distributed archives
remain byte-identical. The certificate records source and table SHA-256
digests and aggregate counts; it contains no participant identifiers.

NLSY79 remains 2,305 participants with observed 2020 cognitive outcomes.
HRS Tracker cannot establish NLSY79 mortality: no independent NLSY mortality
registry audit is claimed. Its 1,845 development / 460 locked-test split and
all predictor lists remain unchanged. Mortality information is used only for
eligibility, never as an input feature. No model refit or locked-test evaluation
was performed for this change.

## Reproduce the audit

From the branch root, using authorized local inputs:

```powershell
python audit_survival.py --stage1 ../upload/01_harmonized_cohort_data_v3.zip --stage2 ../upload/02_model_ready_cohort_pairs_v1.zip --tracker ../data/interim/hrs_official_2022/trk2022tr_r.dta --output metadata/survival_audit.json
python -m part2_models validate --data ../upload/02_model_ready_cohort_pairs_v1.zip
python -m unittest discover -s tests
```

`part2_models/survival.py` is the shared eligibility implementation. The local
raw-processing script `Part2/scripts/22_complete_hrs_alignment_v3.py` now applies
it before setting all three pair eligibility flags. The frozen Stage 02 builder
and every modeling entry point require the passing certificate bound to the
exact HRS CSV hash through `stage2_config.json`. Any future data change requires
a fresh mortality audit and deliberate contract update; it cannot silently
reuse this certificate. Historical artifacts remain historical; the certificate
does not assert that an uninspected UNC copy has been updated.
