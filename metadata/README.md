# Unified metadata

`catalog.json` is the unified catalog for both processing stages. It describes all 34 tables, their ordered columns, keys, record counts, missing counts, checksums, field definitions, source rules, categorical codebooks, and predictor sets. Tables using official CogImp sensitivity scores have an explicit cognition context.

`model_fields.csv` is the readable view of every column in the two modeling tables. It has 103 HRS entries and 185 NLSY79 entries. The 186 distinct names include predictors, outcomes, identifiers, survey weights, split labels, timing information, and quality fields. The `feature_sets` column indicates inclusion in a model input list.

`source_rules.json` preserves the harmonization rules, source question identifiers, common category mappings, available official category labels, and original processing-script checksums. Code labels are indexed by questionnaire identifier and survey year.

`validation.json` records dictionary coverage and checks on the current sample, cognitive totals, category indicators, and NLSY79 predictor measurement years.

## Rebuild

Run from the repository root after obtaining both local archives.

```sh
python build_metadata.py local_data/01_harmonized_cohort_data_v3.zip local_data/02_model_ready_cohort_pairs_v1.zip
```

## Field interpretation

An empty cell represents a missing value. When a categorical parent is missing, all of its encoded indicators remain missing. Category indicators use the `parent__code` naming convention. Source-year columns use `field__source_year` and refer to survey years or recorded observation bounds. Actual follow-up duration comes from the two interview months.

The primary shared model has 29 inputs. The expanded shared model has 77 inputs. The NLSY79 downstream candidate list has 111 inputs before any additional encoding. The 82 NLSY79-only additions describe baseline or earlier life-course information.

HRS uses repeated pairs and fixed household folds. A person may contribute to several reporting intervals. The pair-balancing weight is computed within the training subset separately from the provider's cross-sectional sampling weight.

## Source limitations carried into the analysis

RAND RAEDYRS uses the first nonmissing education report across waves, with a Tracker fallback. The exported file lacks a per-person source-wave indicator. The analysis guide therefore includes an education-excluded sensitivity and requires an additional source-wave audit before making a strict prospective-availability claim for this field.

Depression symptoms and vigorous activity use approximate cross-cohort mappings. Income compares different family units and is an optional sensitivity field. NLSY79 score thresholds represent operational cognitive-score groups. The provider's official imputed and web-adjusted cognition scores are identified in separate sensitivity tables.
