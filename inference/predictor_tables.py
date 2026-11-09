"""Sources for tables of 2000 predictor distributions and baseline missingness."""
import pandas as pd

from inference.importance import predictor_groups
from train.cohorts import ARTIFACTS, atomic_csv, load_cohorts

CONTINUOUS = {"child", "ageym", "height", "weight", "livpar", "momage", "dadage", "livsib",
              "nrstim", "nrsnit", "doctim", "depres", "adl5a", "mobila", "lgmusa", "grossa", "finea"}


def write():
    distributions, summaries = [], []
    for horizon in (6, 8):
        dev, _, predictors = load_cohorts(horizon)
        for variable, columns in predictor_groups(predictors).items():
            for population, frame in (("all", dev), ("Y=0", dev[dev.y.eq(0)]), ("Y=1", dev[dev.y.eq(1)])):
                complete = frame[columns].notna().all(axis=1)
                observed = frame.loc[complete, columns]
                n = len(frame)
                base = {"horizon": horizon, "population": population, "variable": variable,
                        "n": n, "observed": len(observed), "missing": n-len(observed)}
                if len(columns) == 1:
                    series = observed[columns[0]]
                    if variable in CONTINUOUS:
                        summaries.append({**base, "mean": series.mean(), "sd": series.std(),
                                          "median": series.median(), "p25": series.quantile(.25),
                                          "p75": series.quantile(.75), "min": series.min(), "max": series.max()})
                    labels = series
                else:
                    labels = observed.apply(lambda row: next((c for c in columns if row[c] == 1), "reference"), axis=1)
                for level, count in labels.value_counts().items():
                    distributions.append({**base, "level": level, "count": count, "fraction_total": count/n})
                distributions.append({**base, "level": "(missing)", "count": n-len(observed),
                                      "fraction_total": (n-len(observed))/n})
    atomic_csv(pd.DataFrame(summaries), ARTIFACTS / "tables/S9_predictor_summary.csv")
    atomic_csv(pd.DataFrame(distributions), ARTIFACTS / "tables/S10_predictor_distribution.csv")


if __name__ == "__main__":
    write()
