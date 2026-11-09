"""Rebuild Part 1 cohorts from per-wave cognition and the HRS 2022 Tracker."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PRE = ROOT / "data" / "preprocessed data"
WINDOWS = {(2000, 2006): "K", (2000, 2008): "L", (2010, 2016): "P", (2010, 2018): "Q"}
WAVE_PREFIX = dict(zip(range(2000, 2020, 2), "GHJKLMNOPQ"))
NON_PREDICTORS = ("hhidpn", "demcls", "y")


def clean_columns(columns):
    return [c.replace("_", "").replace("/", " or ").replace(".", " ").replace("-", "to")
            .replace("+", "above").replace("0", "").replace(",", " ") for c in columns]


def atomic_csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def build(tracker_path):
    ref = PRE / "2000(00-06y)encode.csv"
    columns = pd.read_csv(ref, nrows=0).columns.tolist()
    predictors = [c for c in columns if c not in NON_PREDICTORS]
    from train.imputation import category_groups
    groups = category_groups(predictors)
    assert len(set(predictors)) == 94
    assert len(predictors) - sum(len(v) - 1 for v in groups.values()) == 60
    tracker_cols = ["HHID", "PN", "KNOWNDECEASEDYR"]
    tracker_cols += [p + "ALIVE" for p in WAVE_PREFIX.values()]
    tracker_cols += [p + "IWWAVE" for p in WINDOWS.values()]
    tracker = pd.read_stata(tracker_path, columns=tracker_cols, convert_categoricals=False)
    tracker["hhidpn"] = pd.to_numeric(tracker.pop("HHID")) * 1000 + pd.to_numeric(tracker.pop("PN"))
    tracker = tracker.set_index("hhidpn", verify_integrity=True)
    flow_path = PRE / "flow_2000.csv"
    raw_2000 = pd.read_csv(flow_path).iloc[0]
    files = [ref, tracker_path, flow_path]
    cognition = {}
    for year in (2002, 2004, 2006, 2008, 2012, 2014, 2016, 2018):
        yy = f"{year % 100:02d}"
        path = PRE / f"_{yy}y.csv"
        files.append(path)
        col = f"{yy}demcls"
        cognition[year] = pd.read_csv(path, usecols=["hhidpn", col]).set_index(
            "hhidpn", verify_integrity=True)[col]
    cohort_dir = ARTIFACTS / "cohorts"
    table_dir = ARTIFACTS / "tables"
    for folder in (cohort_dir, table_dir):
        folder.mkdir(parents=True, exist_ok=True)
    s1 = {}
    flow_rows = []
    for (year, final), prefix in WINDOWS.items():
        path = PRE / f"{year}.csv"
        files.append(path)
        wave = pd.read_csv(path)
        wave.columns = clean_columns(wave.columns)
        assert wave.columns.is_unique
        wave = wave[[c for c in columns if c != "y"]].set_index("hhidpn", verify_integrity=True)
        interviewed = wave["ageym"].notna()
        age = wave["ageym"].gt(50)
        base = wave.loc[age & wave.demcls.eq(0)]
        follow = range(year + 2, final + 1, 2)
        assessments = pd.DataFrame({w: cognition[w].reindex(base.index) for w in follow}, index=base.index)
        assert assessments.isin([0, 1]).where(assessments.notna(), True).all().all()
        vital = tracker.reindex(base.index)
        final_vital = vital[prefix + "ALIVE"]
        death_flags = vital[[WAVE_PREFIX[w] + "ALIVE" for w in follow]].isin([5, 6])
        death = death_flags.any(axis=1) | vital.KNOWNDECEASEDYR.between(1900, final - 1)
        alive = final_vital.eq(1) & ~death
        final_interview = vital[prefix + "IWWAVE"].eq(1)
        observed_case = assessments.eq(1).any(axis=1)
        first_case = assessments.eq(1).idxmax(axis=1).where(observed_case)
        missing_before_case = pd.Series(False, index=base.index)
        for wave_year in list(follow)[1:]:
            missing_before_case |= first_case.eq(wave_year) & assessments.loc[:, :wave_year-2].isna().any(axis=1)
        case = alive & observed_case
        control = alive & ~observed_case & assessments[final].eq(0)
        any_control = alive & ~observed_case & assessments.eq(0).any(axis=1)
        unresolved = ~(case | control | death)
        lost = unresolved & ~final_interview
        status = np.select([case, control, death & observed_case, death, lost],
                           ["incident dementia", "confirmed non-dementia",
                            "death with observed dementia", "death without observed dementia",
                            "loss to follow-up"], default="unresolved after final interview")
        window = f"{year}-{final}"
        ledger = assessments.rename(columns=lambda w: f"dementia_{w}").assign(
            tracker_final_vital=final_vital, tracker_final_interview=final_interview,
            death_by_final_wave=death, alive_at_final_wave=alive, status=status,
            missing_before_first_dementia=missing_before_case)
        for name, negative in (("final", control), ("any", any_control)):
            y = pd.Series(np.select([case, negative], [1, 0], default=np.nan), index=base.index)
            ledger[f"y_{name}"] = y
            cohort = base.assign(y=y).dropna(subset=["y"]).reset_index()[columns]
            cohort["y"] = cohort["y"].astype(int)
            assert not death.loc[cohort.hhidpn].any() and alive.loc[cohort.hhidpn].all()
            atomic_csv(cohort, cohort_dir / f"{window}_{name}.csv")
        atomic_csv(ledger.reset_index(), cohort_dir / f"{window}_audit.csv")
        counts = [len(base), int((case | control).sum()), int(case.sum()),
                  int((case & missing_before_case).sum()), int(control.sum()),
                  int((control & assessments.iloc[:, :-1].isna().any(axis=1)).sum()),
                  int(lost.sum()), int((unresolved & final_interview).sum()), int(death.sum()),
                  int((death & observed_case).sum()), int((death & ~observed_case).sum())]
        assert counts[0] == counts[1] + counts[6] + counts[7] + counts[8]
        assert counts[1] == counts[2] + counts[4] and counts[8] == counts[9] + counts[10]
        s1[window] = counts
        complete = base[predictors].notna().all(axis=1)
        n_interviewed = int(interviewed.sum())
        n_under_age = int((interviewed & ~age).sum())
        n_baseline_cases = int((age & wave.demcls.eq(1)).sum())
        n_baseline_unknown = int((age & ~wave.demcls.isin([0, 1])).sum())
        if year == 2000:
            # The supplied 2000 predictor file was filtered before export.
            assert n_interviewed == raw_2000["remaining"] and n_under_age == n_baseline_cases == 0
            n_interviewed = int(raw_2000["interviewed in 2000"])
            n_under_age = int(raw_2000["excluded: age <= 50"])
            n_baseline_cases = int(raw_2000["excluded: dementia at baseline"])
        assert n_interviewed - n_under_age - n_baseline_cases - n_baseline_unknown == len(base)
        flow_rows.append({"cohort": f"{year} baseline", "horizon (years)": final-year,
                          "interviewed at baseline": n_interviewed,
                          "excluded: age <= 50": n_under_age,
                          "excluded: dementia at baseline": n_baseline_cases,
                          "excluded: baseline dementia status unknown": n_baseline_unknown,
                          "eligible": counts[0], "analytic cohort": counts[1],
                          "incident dementia": counts[2], "confirmed non-dementia": counts[4],
                          "excluded: loss to follow-up": counts[6],
                          "excluded: unresolved after final interview": counts[7],
                          "excluded: died by final wave": counts[8],
                          "excluded: death with observed dementia": counts[9],
                          "excluded: death without observed dementia": counts[10],
                          "complete predictors": int((complete & (case | control)).sum()),
                          "incident dementia, complete predictors": int((complete & case).sum()),
                          "sensitivity analytic cohort": int((case | any_control).sum())})
        print(window, counts, "sensitivity n =", int((case | any_control).sum()), flush=True)
    descriptions = [
        ("Eligible at baseline", "", "total"),
        ("Analytic cohort", "", "total"),
        ("Incident dementia", "Y = 1", ""),
        ("  First dementia observation after a missing intermediate assessment", "Y = 1", "subset"),
        ("Confirmed non-dementia at the final wave", "Y = 0", ""),
        ("  Confirmed at final assessment after a missing intermediate assessment", "Y = 0", "subset"),
        ("Loss to follow-up (no final interview and no assignable outcome)", "Excluded", ""),
        ("Unresolved after final interview", "Excluded", ""),
        ("Died by the final wave", "Excluded", ""),
        ("  With observed dementia", "Excluded", "subset"),
        ("  Without observed dementia", "Excluded", "subset"),
    ]
    s1_frame = pd.DataFrame(descriptions, columns=["Category", "Classification", "Row type"]).assign(**s1)
    atomic_csv(s1_frame, table_dir / "S1_outcome_flow.csv")
    atomic_csv(pd.DataFrame(flow_rows), table_dir / "participant_flow.csv")
    (cohort_dir / "manifest.json").write_text(json.dumps({
        "baseline": "age > 50 and observed dementia-free at baseline",
        "alive": "Final Tracker ALIVE=1 without a death flag (5/6) at any follow-up wave or a known death year preceding the final wave",
        "loss_to_followup": "No final core interview and no assignable outcome, excluding deaths",
        "sensitivity": "Allow an observed dementia-free assessment at any follow-up for confirmed survivors with no observed dementia",
        "predictors": 60, "model_inputs": predictors,
        "sources_sha256": {str(f): sha256(f) for f in dict.fromkeys(files)},
        "cohorts_sha256": {p.name: sha256(p) for p in sorted(cohort_dir.glob("*.csv"))},
    }, indent=2), encoding="utf-8")


def load_cohorts(horizon, outcome="final"):
    if horizon not in (6, 8) or outcome not in ("final", "any"):
        raise ValueError("Unsupported horizon or outcome")
    dev, ext = (pd.read_csv(ARTIFACTS / "cohorts" / f"{year}-{year+horizon}_{outcome}.csv")
                for year in (2000, 2010))
    predictors = [c for c in dev if c not in NON_PREDICTORS]
    assert len(predictors) == 94 and list(dev) == list(ext)
    return dev, ext, predictors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=Path, required=True)
    build(parser.parse_args().tracker)
