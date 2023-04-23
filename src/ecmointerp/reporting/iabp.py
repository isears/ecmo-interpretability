"""
Determine ECMO circuit properties for patients with evidence of ECMO treatment
"""
import dask.dataframe as dd
import pandas as pd
import datetime
from ecmointerp.dataProcessing.util import all_inclusive_dtypes


def get_iabp(sids: pd.Series):
    ecmo_sids = pd.read_csv("cache/ecmo_stayids.csv")["stay_id"]
    all_ce = dd.read_csv("mimiciv/icu/chartevents.csv", dtype=all_inclusive_dtypes)

    iabp_itemids = [
        225335,
        225336,
        225337,
        225338,
        225339,
        225340,
        225341,
        225342,
        225742,
        225778,
        225979,
        225980,
        225981,
        225982,
        225984,
        225985,
        225986,
        225987,
        225988,
        226110,
        227754,
    ]

    iabp_events = all_ce[all_ce["itemid"].isin(iabp_itemids)]
    iabp_events = iabp_events[iabp_events["stay_id"].isin(sids.to_list())].compute(
        scheduler="processes"
    )

    print(f"Counted {iabp_events['stay_id'].nunique()} icu stays w/iabp events")


if __name__ == "__main__":
    sids = pd.read_csv("cache/included_stayids.csv")
    # sids = pd.read_csv("cache/ecmo_stayids.csv")
    print(f"Total n: {len(sids)}")
    get_iabp(sids["stay_id"])
