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

    print("ECMO configurations: n (%)")
    for config_type, count in config_by_sid.value_counts().to_dict().items():
        print(f"{config_type}: {count} ({100* count / len(ecmo_sids):.2f})")
