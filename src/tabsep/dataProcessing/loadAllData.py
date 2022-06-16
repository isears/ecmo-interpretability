"""
Load all data with dask like a wild man
"""
import pandas as pd
import dask
import dask.dataframe as dd
from tabsep.dataProcessing.util import all_inclusive_dtypes
import datetime
import random
from typing import List
from dask.diagnostics import ProgressBar
import sys


class BaseAggregator:
    def _cut(self):
        raise NotImplemented

    def _select_sample(self, labeled_icustays: pd.DataFrame) -> List[int]:
        raise NotImplemented

    def _label(self, df_to_label: pd.DataFrame) -> pd.DataFrame:
        raise NotImplemented

    def __init__(self, prediction_window=12) -> None:
        dask.config.set(scheduler="processes")
        random.seed(42)
        self.minimum_cut = datetime.timedelta(hours=6)
        self.prediction_window = prediction_window

        included_stay_ids = pd.read_csv("cache/included_stayids.csv").squeeze("columns")
        included_features = pd.read_csv("cache/included_features.csv").squeeze(
            "columns"
        )
        print(
            f"Initiated data loading with {len(included_stay_ids)} ICU stays in inclusion criteria"
        )

        print("Randomizing cut times before processing")
        icustays = pd.read_csv(
            "mimiciv/icu/icustays.csv",
            usecols=["stay_id", "hadm_id", "subject_id", "intime", "outtime"],
            parse_dates=["intime", "outtime"],
        )

        sepsis_df = pd.read_csv(
            "mimiciv/derived/sepsis3.csv",
            parse_dates=[
                "sofa_time",
                "suspected_infection_time",
                "culture_time",
                "antibiotic_time",
            ],
        )

        # Filter by inclusion criteria
        icustays = icustays[icustays["stay_id"].isin(included_stay_ids)]
        sepsis_df = sepsis_df[sepsis_df["stay_id"].isin(included_stay_ids)]

        sepsis_df["sepsis_time"] = sepsis_df.apply(
            lambda row: max(row["sofa_time"], row["suspected_infection_time"]), axis=1
        )

        sepsis_df["sepsis3"] = 1.0

        icustays = icustays.merge(
            sepsis_df[["stay_id", "sepsis_time", "sepsis3"]], how="left", on="stay_id"
        )

        icustays["sepsis3"] = icustays["sepsis3"].fillna(0.0)

        sample_stay_ids = self._select_sample(icustays)

        # Filter by sample
        icustays = icustays[icustays["stay_id"].isin(sample_stay_ids)]

        print(f"[{self.__class__.__name__}] Sampling {len(icustays)}")

        icustays["cut_time"] = icustays.apply(self._cut, axis=1)
        icustays["cut_los"] = icustays["cut_time"] - icustays["intime"]
        icustays["cut_los"] = icustays["cut_los"].apply(lambda x: x.total_seconds())

        icustays = self._label(icustays)

        # Check work
        icustays["sepsis_time"] = icustays["sepsis_time"].fillna(datetime.datetime.max)
        assert (icustays["cut_time"] < icustays["sepsis_time"]).all()
        assert (icustays["cut_time"] > icustays["intime"] + self.minimum_cut).all()

        self.icustays = icustays
        self.metadata = icustays[["stay_id", "cut_los", "label"]]
        self.sample_stay_ids = sample_stay_ids
        self.included_features = included_features
        print("[+] Cut times randomized, running processor...")

    def agg_general(self, df_in, agg_fn: str):

        # filter inclusion criteria
        df_in = df_in[df_in["stay_id"].isin(self.sample_stay_ids)]
        df_in = df_in[df_in["itemid"].isin(self.included_features)]
        df_in = df_in.dropna(how="any")

        df_in = df_in.merge(
            self.icustays[["stay_id", "cut_time"]], how="left", on="stay_id"
        )

        df_in = df_in[df_in["cut_time"].gt(df_in["charttime"])]

        avg_vals = (
            df_in[["stay_id", "itemid", "valuenum"]]
            .groupby(["stay_id", "itemid"])
            .apply(agg_fn, meta=pd.DataFrame({"valuenum": [0.0]}))
            .compute()  # Dask doesn't implement unstack, so have to compute here
            .unstack(fill_value=0.0)
        )

        # set new columns to be itemids
        avg_vals.columns = avg_vals.columns.map(lambda col: col[1])

        def get_max_event(g):
            if len(g) == 0:
                return pd.NA
            else:
                return g.loc[g["charttime"].idxmax()]["valuenum"]

        latest_vals = (
            df_in[["stay_id", "itemid", "charttime", "valuenum"]]
            .groupby(["stay_id", "itemid"])
            .apply(get_max_event, meta=pd.Series([0.0]))
            .compute()
            .unstack(fill_value=0.0)
        )

        combined_vals = avg_vals.merge(
            latest_vals, left_index=True, right_index=True, suffixes=("_avg", "_latest")
        )

        # Add in labels
        combined_vals = combined_vals.merge(
            self.metadata,
            how="left",
            left_index=True,
            right_on="stay_id",
        )

        return combined_vals

    def agg_chartevents(self):
        chartevents = dd.read_csv(
            "mimiciv/icu/chartevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "valuenum", "charttime"],
            parse_dates=["charttime"],
        )

        return self.agg_general(chartevents, "mean")

    def agg_outputevents(self):
        outputevents = dd.read_csv(
            "mimiciv/icu/outputevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "value", "charttime"],
            parse_dates=["charttime"],
        )

        outputevents = outputevents.rename(columns={"value": "valuenum"})

        return self.agg_general(outputevents, "sum")

    def agg_inputevents(self):
        inputevents = dd.read_csv(
            "mimiciv/icu/inputevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "starttime", "rate"],
            parse_dates=["starttime"],
        )

        inputevents = inputevents.rename(
            columns={"rate": "valuenum", "starttime": "charttime"}
        )

        return self.agg_general(inputevents, "mean")

    def agg_procedureevents(self):
        procedureevents = dd.read_csv(
            "mimiciv/icu/procedureevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "starttime", "value"],
            parse_dates=["starttime"],
        )

        procedureevents = procedureevents.rename(
            columns={"value": "valuenum", "starttime": "charttime"}
        )

        return self.agg_general(procedureevents, "sum")

    def agg_static(self):
        admissions = pd.read_csv("mimiciv/core/admissions.csv")

        # Need label for non-unique stayid merging
        static_df = self.icustays[["stay_id", "hadm_id", "label", "cut_los"]].merge(
            admissions[
                [
                    "hadm_id",
                    "admission_type",
                    "admission_location",
                    "insurance",
                    "language",
                    "marital_status",
                    "ethnicity",
                ]
            ],
            how="left",
            on="hadm_id",
        )

        static_df = static_df.drop(columns=["hadm_id"])
        static_df = pd.get_dummies(static_df)

        # Just extract age from patients
        patients_df = pd.read_csv("mimiciv/core/patients.csv")
        other_static = self.icustays.merge(
            patients_df[["anchor_age", "anchor_year", "subject_id", "gender"]],
            how="left",
            on="subject_id",
        )

        other_static["age_on_admission"] = other_static.apply(
            lambda row: row["intime"].year - row["anchor_year"] + row["anchor_age"],
            axis=1,
        )

        other_static["gender"] = other_static["gender"].apply(
            lambda x: 1.0 if x == "M" else 0.0
        )

        static_df = static_df.merge(
            other_static[["gender", "stay_id", "age_on_admission"]],
            how="left",
            on="stay_id",
        )

        return static_df

    def agg_all(self):
        # TODO: replace load_from_disk
        raise NotImplemented


class NoSepsisAggregator(BaseAggregator):
    """
    Selects a sample of icu stays with no sepsis, and then,
    within that, selects a random truncation point in the icu stay
    """

    def _select_sample(self, labeled_icustays: pd.DataFrame) -> List[int]:
        sepsis_count = int(labeled_icustays["sepsis3"].sum())
        no_sepsis_stays = labeled_icustays[labeled_icustays["sepsis3"] == 0]
        # Balance non-septic icu stays with septic icu stays
        return (
            no_sepsis_stays["stay_id"].sample(n=sepsis_count, random_state=42).to_list()
        )

    def _cut(self, row):
        delta = row["outtime"] - row["intime"]
        # Always ensure at least min_cut (e.g. 6 hours) of ICU time
        random_index_second = random.randrange(
            self.minimum_cut.total_seconds(), delta.total_seconds()
        )
        return row["intime"] + datetime.timedelta(seconds=random_index_second)

    def _label(self, df_to_label: pd.DataFrame) -> pd.DataFrame:
        df_to_label["label"] = 0.0
        return df_to_label

    def __init__(self) -> None:
        super().__init__()


class SepsisPosAggregator(BaseAggregator):
    """
    Selects a sample of icu stays with sepsis and then truncates icu stay 12 hrs. before sepsis onset
    """

    def _select_sample(self, labeled_icustays: pd.DataFrame) -> List[int]:
        sepsis_stays = labeled_icustays[labeled_icustays["sepsis3"] == 1]
        # Balance non-septic icu stays with septic icu stays
        return sepsis_stays["stay_id"].to_list()

    def _cut(self, row):
        return row["sepsis_time"] - datetime.timedelta(hours=self.prediction_window)

    def _label(self, df_to_label: pd.DataFrame) -> pd.DataFrame:
        df_to_label["label"] = 1.0  # All cases will be positive
        return df_to_label

    def __init__(self) -> None:
        super().__init__()


class SepsisNegAggregator(SepsisPosAggregator):
    """
    Selects a sample of icu stays with sepsis then truncates icu stay between
    6hrs. after beginning of stay to  11 hrs. before sepsis onset
    """

    def _cut(self, row):
        delta = (
            # Give 1 hr. before the prediction window to differentiate a bit
            row["sepsis_time"]
            - datetime.timedelta(hours=self.prediction_window - 1)
        ) - row["intime"]

        random_index_second = random.randrange(
            self.minimum_cut.total_seconds(), delta.total_seconds()
        )
        return row["intime"] + datetime.timedelta(seconds=random_index_second)

    def _label(self, df_to_label: pd.DataFrame) -> pd.DataFrame:
        df_to_label["label"] = 0.0
        return df_to_label

    def __init__(self) -> None:
        super().__init__()


def meta_loader(version_name: str) -> pd.DataFrame:
    # TODO: could clean this up a bit
    ProgressBar().register()

    all_ie = list()
    all_ce = list()
    all_pe = list()
    all_static = list()

    for agg_cls in [NoSepsisAggregator, SepsisPosAggregator, SepsisNegAggregator]:
        curr_aggregator = agg_cls()

        print("Agg'ing inputevents")
        processed_inputevents = curr_aggregator.agg_inputevents()
        all_ie.append(processed_inputevents)

        print("Agg'ing chartevents")
        processed_chartevents = curr_aggregator.agg_chartevents()
        all_ce.append(processed_chartevents)

        print("Agg'ing procedureevents")
        processed_procedureevents = curr_aggregator.agg_procedureevents()
        all_pe.append(processed_procedureevents)

        print("Agg'ing static")
        static = curr_aggregator.agg_static()
        all_static.append(static)

    combined_df = pd.concat(all_ie)
    combined_df = combined_df.merge(
        pd.concat(all_ce), how="left", on=["stay_id", "cut_los", "label"]
    )
    combined_df = combined_df.merge(
        pd.concat(all_pe), how="left", on=["stay_id", "cut_los", "label"]
    )
    combined_df = combined_df.merge(
        pd.concat(all_static), how="left", on=["stay_id", "cut_los", "label"]
    )

    combined_df = combined_df.fillna(0.0)
    combined_df.to_csv(f"cache/{version_name}.csv", index=False)


def load_from_disk() -> pd.DataFrame:
    df_out = pd.read_csv(
        "cache/processed_metadata.csv", index_col="stay_id", low_memory=False
    )

    for csv in [
        "cache/processed_chartevents.csv",
        "cache/processed_inputevents.csv",
        "cache/processed_outputevents.csv",
        "cache/processed_procedureevents.csv",
        "cache/processed_static.csv",
    ]:
        df = pd.read_csv(csv, index_col="stay_id", low_memory=False)
        df_out = df_out.merge(df, how="left", left_index=True, right_index=True)

    df_out = df_out.fillna(0.0)

    return df_out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        version_name = sys.argv[1]
    else:
        version_name = "processed_combined"

    meta_loader(version_name)
