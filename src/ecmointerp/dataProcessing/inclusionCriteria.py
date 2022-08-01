"""
Apply inclusion criteria to generate a list of included stay ids
"""
import pandas as pd
import dask.dataframe as dd
import datetime


class InclusionCriteria:
    def __init__(self):
        self.all_stays = pd.read_csv(
            "mimiciv/icu/icustays.csv",
            usecols=["stay_id", "hadm_id", "intime", "outtime"],
            dtype={
                "stay_id": "int",
                "hadm_id": "int",
                "intime": "str",
                "outtime": "str",
            },
            parse_dates=["intime", "outtime"],
        )

        self.admissions = pd.read_csv(
            "mimiciv/core/admissions.csv",
            dtype={
                "subject_id": "int",
                "hadm_id": "int",
                "admittime": "str",
            },
            parse_dates=["admittime"],
        )

        self.patients = pd.read_csv(
            "mimiciv/core/patients.csv",
            dtype={"subject_id": "int", "anchor_age": "int", "anchor_year": "int"},
        )

        self.admission_patients = self.admissions.merge(
            self.patients, how="left", on="subject_id"
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

    def _exclude_double_stays(self):
        self.all_stays = self.all_stays[
            self.all_stays.groupby("hadm_id")["hadm_id"].transform("size") == 1
        ]

    def _exclude_under_18(self):
        self.admission_patients["zero_year"] = (
            self.admission_patients["anchor_year"]
            - self.admission_patients["anchor_age"]
        )
        self.admission_patients["age_at_admission"] = (
            self.admission_patients["admittime"].dt.year
            - self.admission_patients["zero_year"]
        )
        over_18_admissions = self.admission_patients[
            self.admission_patients["age_at_admission"] > 18
        ]

        self.all_stays = self.all_stays[
            self.all_stays["hadm_id"].isin(over_18_admissions["hadm_id"])
        ]

    def _exclude_long_stays(self, time_hours=(24 * 30)):
        self.all_stays = self.all_stays[
            self.all_stays.apply(
                lambda row: (row["outtime"] - row["intime"])
                < datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]

    def _get_ecmo_sids(self):
        chartevents_dd = dd.read_csv(
            "mimiciv/icu/chartevents.csv",
            assume_missing=True,
            blocksize=100e6,
            dtype={
                "subject_id": "int",
                "hadm_id": "int",
                "stay_id": "int",
                "charttime": "object",
                "storetime": "object",
                "itemid": "int",
                "value": "object",
                "valueuom": "object",
                "warning": "object",
                "valuenum": "float",
            },
        )

        ecmo_circuit_config_ids = [229268, 229840]
        ecmo_flow_ids = [229270, 229842]

        # Most restrictive filtering: must have a configuration event and at least one flow measurement
        ecmo_events = chartevents_dd[
            chartevents_dd["itemid"].isin(ecmo_flow_ids + ecmo_circuit_config_ids)
        ].compute(scheduler="processes")

        ecmo_by_stay = ecmo_events.groupby("stay_id", as_index=False).agg(
            has_config=("itemid", lambda x: x.isin(ecmo_circuit_config_ids).any()),
            has_flow=("itemid", lambda x: x.isin(ecmo_flow_ids).any()),
        )

        config_and_flow = ecmo_by_stay[
            ecmo_by_stay["has_config"] & ecmo_by_stay["has_flow"]
        ]

        return config_and_flow["stay_id"].to_list()

    def get_included(self):
        order = [
            self._exclude_nodata,
            self._exclude_double_stays,
            self._exclude_under_18,
            self._exclude_long_stays,
        ]

        for func in order:
            count_before = len(self.all_stays)
            func()
            count_after = len(self.all_stays)
            print(f"{func.__name__} excluded {count_before - count_after} stay ids")

        # this takes 4- or maybe even 5-ever
        print("Finding ECMO ids, this will take some time...")
        ecmo_sids = self._get_ecmo_sids()
        print(f"Found {len(ecmo_sids)}")

        base_dataset = self.all_stays[~self.all_stays["stay_id"].isin(ecmo_sids)]
        ecmo_dataset = self.all_stays[self.all_stays["stay_id"].isin(ecmo_sids)]

        print(f"Final size of base dataset: {len(base_dataset)}")
        print(f"Final size of ecmo dataset: {len(ecmo_dataset)}")

        base_dataset["stay_id"].to_csv("cache/included_stayids.csv", index=False)
        ecmo_dataset["stay_id"].to_csv("cache/ecmo_stayids.csv", index=False)
        return self.all_stays["stay_id"].to_list()


if __name__ == "__main__":
    ic = InclusionCriteria()
    stay_ids = ic.get_included()
