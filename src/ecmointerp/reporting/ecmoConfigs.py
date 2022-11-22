"""
Determine ECMO circuit properties for patients with evidence of ECMO treatment
"""
import dask.dataframe as dd
import pandas as pd
from ecmointerp.dataProcessing.util import all_inclusive_dtypes


if __name__ == "__main__":
    ecmo_sids = pd.read_csv("cache/ecmo_stayids.csv")["stay_id"]
    all_ce = dd.read_csv("mimiciv/icu/chartevents.csv", dtype=all_inclusive_dtypes)

    all_ce = all_ce[all_ce["stay_id"].isin(ecmo_sids.to_list())].compute(
        scheduler="processes"
    )

    all_ce["charttime"] = pd.to_datetime(all_ce["charttime"])

    config_events = all_ce[all_ce["itemid"].isin([229268, 229840])]

    def get_config(config_history: pd.Series):
        if config_history.nunique() > 1:  # Should only have VV or VA in mimic
            allconfigs = config_history.unique()
            assert "VV" in allconfigs
            assert "VA" in allconfigs
            assert len(allconfigs) == 2
            return "Both"
        else:
            assert len(config_history.unique())
            return config_history.unique()[0]

    config_by_sid = config_events.groupby("stay_id")["value"].apply(get_config)

    # Configs
    print("ECMO configurations: n (%)")
    for config_type, count in config_by_sid.value_counts().to_dict().items():
        print(f"{config_type}: {count} ({100* count / len(ecmo_sids):.2f})")

    # Duration of therapy
    print("ECMO duration: mean (median +/- std)")
    flow_events = all_ce[all_ce["itemid"].isin([229270, 229842])]
    max_flowtimes = flow_events.groupby("stay_id")["charttime"].apply(max)
    min_flowtimes = flow_events.groupby("stay_id")["charttime"].apply(min)

    max_flowtimes.name = "ecmo_end"
    min_flowtimes.name = "ecmo_start"

    ecmo_times = pd.merge(
        max_flowtimes, min_flowtimes, how="left", right_index=True, left_index=True
    )
    assert len(ecmo_times) == len(ecmo_sids)

    ecmo_times["duration_days"] = ecmo_times["ecmo_end"] - ecmo_times["ecmo_start"]
    ecmo_times["duration_days"] = ecmo_times["duration_days"].apply(
        lambda x: x.total_seconds() / (60 * 60 * 24)
    )

    avg = ecmo_times["duration_days"].mean()
    median = ecmo_times["duration_days"].median()
    std = ecmo_times["duration_days"].std()

    print(f"{avg:.2f} ({median:.2f} +/- {std:.2f}")

    # Pre-ecmo ventilation
    print("Ventilation duration pre-ECMO")
    vent_df = pd.read_csv("mimiciv/derived/ventilation.csv")
    # Only mechanical
    vent_df = vent_df[
        vent_df["ventilation_status"].isin(["InvasiveVent", "Tracheostomy"])
    ]

    vent_df = vent_df[vent_df["stay_id"].isin(ecmo_sids)]

    for tc in ["starttime", "endtime"]:
        vent_df[tc] = pd.to_datetime(vent_df[tc])

    vent_df = pd.merge(
        vent_df, ecmo_times, how="left", left_on="stay_id", right_index=True
    )

    # Drop ventilations that started after ECMO treatment
    vent_df = vent_df[~(vent_df["starttime"] > vent_df["ecmo_start"])]

    def get_duration(row):
        diff_seconds = (row["endtime"] - row["starttime"]).total_seconds()
        return diff_seconds / (60 * 60 * 24)

    vent_df["duration_days"] = vent_df.apply(get_duration, axis=1)
    summed_durations = vent_df.groupby("stay_id")["duration_days"].sum()

    avg = summed_durations.mean()
    median = summed_durations.median()
    std = summed_durations.std()

    print(f"{avg:.2f} ({median:.2f} +/- {std:.2f}")

