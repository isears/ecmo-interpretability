import dask.dataframe as dd
import pandas as pd
from tabsep.dataProcessing.util import all_inclusive_dtypes


def generate_missingness(threshold: float) -> pd.Series:
    out = pd.Series([], dtype="int")

    for ds in [
        "mimiciv/icu/chartevents.csv",
        "mimiciv/icu/outputevents.csv",
        "mimiciv/icu/inputevents.csv",
        "mimiciv/icu/procedureevents.csv",
    ]:
        events_dd = dd.read_csv(
            ds,
            assume_missing=True,
            blocksize=100e6,
            usecols=["itemid", "stay_id"],
            dtype=all_inclusive_dtypes,
        )

        total_stays = events_dd["stay_id"].nunique()
        stay_counts_by_itemid = (
            events_dd.groupby("itemid")["stay_id"]
            .nunique()
            .compute(scheduler="processes")
        )

        stay_counts_by_itemid = stay_counts_by_itemid.reset_index()
        stay_counts_by_itemid = stay_counts_by_itemid.rename(
            columns={"stay_id": "stay_count"}
        )
        included_itemids = stay_counts_by_itemid[
            (stay_counts_by_itemid["stay_count"] / total_stays) > threshold
        ]

        print(
            f"Dropped {len(stay_counts_by_itemid) - len(included_itemids)} features that did not meet the missingness threshold ({threshold}) for datasource {ds}"
        )

        out = pd.concat([out, included_itemids["itemid"]])

    print(f"Final feature count: {len(out)}")
    out.to_csv("cache/included_features.csv", index=False)


if __name__ == "__main__":
    generate_missingness(0.1)
