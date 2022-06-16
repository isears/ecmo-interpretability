"""
Apply inclusion criteria to generate a list of included stay ids
"""
import pandas as pd
import dask.dataframe as dd
from tabsep.dataProcessing.util import all_inclusive_dtypes
import datetime


class InclusionCriteria:
    def __init__(self):
        self.all_stays = pd.read_csv(
            "mimiciv/icu/icustays.csv",
            usecols=["stay_id", "intime", "outtime"],
            dtype={"stay_id": "int", "intime": "str", "outtime": "str"},
            parse_dates=["intime", "outtime"],
        )

    def _exclude_nodata(self):
        """
        Exclude patients w/no chartevents
        """
        chartevents = dd.read_csv(
            "mimiciv/icu/chartevents.csv",
            usecols=["stay_id", "subject_id"],
            blocksize=100e6,
        )

        chartevents_stay_ids = (
            chartevents["stay_id"].unique().compute(scheduler="processes")
        )
        self.all_stays = self.all_stays[
            self.all_stays["stay_id"].isin(chartevents_stay_ids)
        ]

    def _exclude_short_stays(self, time_hours=24):
        self.all_stays = self.all_stays[
            self.all_stays.apply(
                lambda row: (row["outtime"] - row["intime"])
                > datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]

    def _exclude_long_stays(self, time_hours=(24 * 30)):
        self.all_stays = self.all_stays[
            self.all_stays.apply(
                lambda row: (row["outtime"] - row["intime"])
                < datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]

    def _exclude_early_sepsis(self, time_hours=24):
        """
        Exclude patients that arrive to the ICU qualifying for sepsis3
        """
        sepsis_df = pd.read_csv(
            "mimiciv/derived/sepsis3.csv",
            parse_dates=[
                "sofa_time",
                "suspected_infection_time",
                "culture_time",
                "antibiotic_time",
            ],
        )
        sepsis_df["sepsis_time"] = sepsis_df.apply(
            lambda row: max(row["sofa_time"], row["suspected_infection_time"]), axis=1
        )

        sepsis_df = sepsis_df.merge(
            self.all_stays[["stay_id", "intime"]], how="left", on="stay_id"
        )

        early_sepsis_stays = sepsis_df[
            sepsis_df.apply(
                lambda row: row["sepsis_time"] - row["intime"]
                < datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]["stay_id"]

        self.all_stays = self.all_stays[
            ~self.all_stays["stay_id"].isin(early_sepsis_stays)
        ]

    def get_included(self):
        order = [
            self._exclude_nodata,
            self._exclude_short_stays,
            self._exclude_long_stays,
            self._exclude_early_sepsis,
        ]

        for func in order:
            count_before = len(self.all_stays)
            func()
            count_after = len(self.all_stays)
            print(f"{func.__name__} excluded {count_before - count_after} stay ids")

        print(f"Saving remaining {len(self.all_stays)} stay ids to disk")
        self.all_stays["stay_id"].to_csv("cache/included_stayids.csv", index=False)
        return self.all_stays["stay_id"].to_list()


if __name__ == "__main__":
    ic = InclusionCriteria()
    stay_ids = ic.get_included()
