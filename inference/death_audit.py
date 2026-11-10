"""Cross-check accepted outcomes against raw HRS Tracker vital and date records."""
import argparse
from pathlib import Path

import pandas as pd

from train.cohorts import ARTIFACTS, TRACKER, WAVE_PREFIX, WINDOWS, atomic_csv


def audit(tracker_path):
    columns = ["HHID", "PN", "KNOWNDECEASEDYR", "KNOWNDECEASEDMO"]
    columns += [p + "ALIVE" for p in WAVE_PREFIX.values()]
    columns += [p + s for p in WINDOWS.values() for s in ("IWYEAR", "IWMONTH")]
    tracker = pd.read_stata(tracker_path, columns=columns, convert_categoricals=False)
    tracker["hhidpn"] = pd.to_numeric(tracker.HHID) * 1000 + pd.to_numeric(tracker.PN)
    tracker = tracker.set_index("hhidpn", verify_integrity=True)
    rows = []
    for (baseline, final), p in WINDOWS.items():
        window = f"{baseline}-{final}"
        labels = pd.read_csv(ARTIFACTS / "cohorts" / f"{window}_audit.csv").set_index("hhidpn")
        vital = tracker.reindex(labels.index)
        waves = [WAVE_PREFIX[y] + "ALIVE" for y in range(baseline, final + 1, 2)]
        wave_death = vital[waves].isin([5, 6]).any(axis=1)
        earlier_year = vital.KNOWNDECEASEDYR.between(1900, final - 1)
        dated = vital.KNOWNDECEASEDMO.between(1, 12) & vital[p + "IWMONTH"].between(1, 12)
        death_month = vital.KNOWNDECEASEDYR * 12 + vital.KNOWNDECEASEDMO
        interview_month = vital[p + "IWYEAR"] * 12 + vital[p + "IWMONTH"]
        death_before_interview = dated & death_month.lt(interview_month)
        assessed = labels.filter(regex=r"^dementia_")
        ever_case = assessed.eq(1).any(axis=1)
        ever_free = assessed.eq(0).any(axis=1)
        for outcome in ("final", "any"):
            accepted = labels[f"y_{outcome}"].notna()
            violations = accepted & (wave_death | earlier_year | death_before_interview)
            if violations.any():
                raise ValueError(f"{window}/{outcome}: {int(violations.sum())} accepted labels with death evidence")
            death = labels.death_by_final_wave
            rows.append({"window": window, "outcome": outcome, "eligible": len(labels),
                         "included": int(accepted.sum()), "deaths_excluded": int(death.sum()),
                         "deaths_with_observed_dementia": int((death & ever_case).sum()),
                         "deaths_without_dementia_with_prior_negative": int((death & ~ever_case & ever_free).sum()),
                         "accepted_with_wave_death_flag": int((accepted & wave_death).sum()),
                         "accepted_with_prior_death_year": int((accepted & earlier_year).sum()),
                         "accepted_with_death_month_before_final_interview": int((accepted & death_before_interview).sum()),
                         "accepted_same_interview_and_death_month_tracker_alive": int((accepted & dated & death_month.eq(interview_month)).sum())})
    result = pd.DataFrame(rows)
    atomic_csv(result, ARTIFACTS / "tables" / "death_exclusion_audit.csv")
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=Path, default=TRACKER)
    audit(parser.parse_args().tracker)
