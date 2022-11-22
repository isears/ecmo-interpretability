from dataclasses import dataclass
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import json


class Table1Generator(object):
    def __init__(self, stay_ids: List[int]) -> None:
        self.stay_ids = stay_ids
        self.table1 = pd.DataFrame(columns=["Item", "Value"])

        self.all_df = pd.read_csv("mimiciv/icu/icustays.csv")
        self.all_df = self.all_df[self.all_df["stay_id"].isin(self.stay_ids)]
        self.total_stays = len(self.all_df.index)

        # Create df with all demographic data
        self.all_df = self.all_df.merge(
            pd.read_csv("mimiciv/derived/sepsis3.csv"),
            how="left",
            on=["stay_id", "subject_id"],
        )

        self.all_df = self.all_df.merge(
            pd.read_csv("mimiciv/core/patients.csv"), how="left", on=["subject_id"]
        )

        self.all_df = self.all_df.merge(
            pd.read_csv("mimiciv/core/admissions.csv"),
            how="left",
            on=["hadm_id", "subject_id"],
        )

        diagnoses_icd = pd.read_csv("mimiciv/hosp/diagnoses_icd.csv")
        diagnoses_icd = (
            diagnoses_icd[["hadm_id", "icd_code"]]
            .groupby("hadm_id")["icd_code"]
            .apply(list)
        )

        self.all_df = self.all_df.merge(
            diagnoses_icd, how="left", left_on="hadm_id", right_index=True
        )

        # Replace nans
        self.all_df["icd_code"] = self.all_df["icd_code"].apply(
            lambda x: [] if type(x) != list else x
        )

        time_columns = [
            "intime",
            "outtime",
            "sofa_time",
            "culture_time",
            "antibiotic_time",
            "suspected_infection_time",
            "dod",
            "admittime",
            "dischtime",
            "deathtime",
            "edregtime",
            "edouttime",
        ]

        for tc in time_columns:
            self.all_df[tc] = pd.to_datetime(self.all_df[tc])

        # Make sure there's only one stay id per entry so we can confidently calculate statistics
        assert len(self.all_df["stay_id"]) == self.all_df["stay_id"].nunique()

    def _add_table_row(self, item: str, value: str):
        self.table1.loc[len(self.table1.index)] = [item, value]

    def _pprint_percent(self, n: int, total: int = None) -> str:
        if total == None:
            total = self.total_stays

        return f"{n}, ({n / total * 100:.2f} %)"

    def _pprint_mean(self, values: pd.Series):
        return f"{values.mean():.2f} (median {values.median():.2f}, std {values.std():.2f})"

    def _tablegen_sepsis(self) -> None:
        # Count sepsis
        sepsis_count = len(self.all_df[self.all_df["sepsis3"] == True])

        self._add_table_row(
            item="Sepsis Prevalence (Sepsis3)", value=self._pprint_percent(sepsis_count)
        )

        # Calculate average time of sepsis onset, as a percentage of LOS
        sepsis_only = self.all_df[self.all_df["sepsis3"] == True]

        sepsis_only["sepsis_time"] = sepsis_only.apply(
            lambda row: max(row["sofa_time"], row["suspected_infection_time"]), axis=1
        )

        sepsis_only["sepsis_percent_los"] = (
            sepsis_only["sepsis_time"] - sepsis_only["intime"]
        ) / (sepsis_only["outtime"] - sepsis_only["intime"])

        self._add_table_row(
            item="Average % of Length of Stay of Sepsis Onset",
            value=self._pprint_mean(sepsis_only["sepsis_percent_los"] * 100),
        )

        sepsis_only["sepsis_timedelta_hours"] = sepsis_only.apply(
            lambda row: (row["sepsis_time"] - row["intime"]).total_seconds()
            / (60 * 60),
            axis=1,
        )

        self._add_table_row(
            item="Average Time of Sepsis Onset after ICU Admission (hrs)",
            value=self._pprint_mean(sepsis_only["sepsis_timedelta_hours"]),
        )

        self._add_table_row(
            item="Septic ICU Stays with Sepsis Onset > 24 hrs after Admission",
            value=self._pprint_percent(
                n=len(sepsis_only[sepsis_only["sepsis_timedelta_hours"] > 24]),
                total=len(sepsis_only),
            ),
        )

    def _tablegen_general_demographics(self) -> None:
        for demographic_name in [
            "gender",
            "ethnicity",
            "marital_status",
            "insurance",
            "admission_type",
            "language",
            "hospital_expire_flag",
        ]:
            for key, value in (
                self.all_df[demographic_name].value_counts().to_dict().items()
            ):
                self._add_table_row(
                    f"[{demographic_name}] {key}", self._pprint_percent(value)
                )

    def _tablegen_age(self) -> None:
        self.all_df["age_at_intime"] = self.all_df.apply(
            lambda row: (
                ((row["intime"].year) - row["anchor_year"]) + row["anchor_age"]
            ),
            axis=1,
        )

        self._add_table_row(
            item="Average Age at ICU Admission",
            value=self._pprint_mean(self.all_df["age_at_intime"]),
        )

    def _tablegen_comorbidities(self) -> None:
        cci = pd.read_csv("mimiciv/derived/cci.csv")
        # cci = cci[cci["hadm_id"].isin(self.all_df["hadm_id"])]

        merged_df = pd.merge(
            self.all_df[["hadm_id", "stay_id"]], cci, how="left", on="hadm_id"
        )
        comorbidities = [
            c
            for c in cci.columns
            if c
            not in ["subject_id", "hadm_id", "age_score", "charlson_comorbidity_index"]
        ]

        for c in comorbidities:
            c_count = len(merged_df[merged_df[c] > 0])
            self._add_table_row(
                item=f"[COMORBIDITY] {c}", value=self._pprint_percent(c_count)
            )

        self._add_table_row(
            item="Avg. CCI",
            value=self._pprint_mean(merged_df["charlson_comorbidity_index"]),
        )

    def _tablegen_bmi(self) -> None:
        self.all_df = self.all_df.merge(
            pd.read_csv("mimiciv/derived/first_day_height.csv"),
            how="left",
            on=["stay_id", "subject_id"],
        )

        self.all_df = self.all_df.merge(
            pd.read_csv("mimiciv/derived/first_day_weight.csv"),
            how="left",
            on=["stay_id", "subject_id"],
        )

        self.all_df["bmi"] = (
            self.all_df["weight_admit"] / (self.all_df["height"] / 100) ** 2
        )

        self._add_table_row(
            item="BMI (kg / m2)", value=self._pprint_mean(self.all_df["bmi"].dropna())
        )

        self._add_table_row(
            item="BMI Missing",
            value=self._pprint_percent(len(self.all_df[self.all_df["bmi"].isna()])),
        )

    def _tablegen_interventions(self) -> None:
        # Vasopressors: norepinephrine, phenylephrine, epinephrine, vasopressin
        norepi_sids = pd.read_csv("mimiciv/derived/norepinephrine.csv")[
            "stay_id"
        ].drop_duplicates()
        phenylephrine_sids = pd.read_csv("mimiciv/derived/phenylephrine.csv")[
            "stay_id"
        ].drop_duplicates()
        epinephrine_sids = pd.read_csv("mimiciv/derived/epinephrine.csv")[
            "stay_id"
        ].drop_duplicates()
        vasopressin_sids = pd.read_csv("mimiciv/derived/vasopressin.csv")[
            "stay_id"
        ].drop_duplicates()

        pressors = pd.concat(
            [norepi_sids, phenylephrine_sids, epinephrine_sids, vasopressin_sids],
            ignore_index=True,
        ).drop_duplicates()

        pressors_subset = [
            sid for sid in self.all_df["stay_id"].to_list() if sid in pressors.to_list()
        ]

        self._add_table_row(
            item="Vasopressors", value=self._pprint_percent(len(pressors_subset))
        )

        # Inotropes: epinephrine, dopamine, dobutamine, milrinone
        dopamine_sids = pd.read_csv("mimiciv/derived/dopamine.csv")[
            "stay_id"
        ].drop_duplicates()
        dobutamine_sids = pd.read_csv("mimiciv/derived/dobutamine.csv")[
            "stay_id"
        ].drop_duplicates()
        milrinone_sids = pd.read_csv("mimiciv/derived/milrinone.csv")[
            "stay_id"
        ].drop_duplicates()

        inotropes = pd.concat(
            [epinephrine_sids, dobutamine_sids, dopamine_sids, milrinone_sids],
            ignore_index=True,
        ).drop_duplicates()

        inotropes_subset = [
            sid
            for sid in self.all_df["stay_id"].to_list()
            if sid in inotropes.to_list()
        ]

        self._add_table_row(
            item="Inotropes", value=self._pprint_percent(len(inotropes_subset))
        )

        # Paralysis
        paralysis_sids = pd.read_csv("mimiciv/derived/neuroblock.csv")[
            "stay_id"
        ].drop_duplicates()

        paralysis_subset = [
            sid
            for sid in self.all_df["stay_id"].to_list()
            if sid in paralysis_sids.to_list()
        ]

        self._add_table_row(
            item="Paralysis", value=self._pprint_percent(len(paralysis_subset))
        )

        # Renal replacement therapy
        rrt_sids = pd.read_csv("mimiciv/derived/rrt.csv")["stay_id"].drop_duplicates()

        rrt_subset = [
            sid for sid in self.all_df["stay_id"].to_list() if sid in rrt_sids.to_list()
        ]

        self._add_table_row(
            item="Renal Replacement Therapy",
            value=self._pprint_percent(len(rrt_subset)),
        )

    def _tablegen_icds(self) -> None:
        # Intra-aortic balloon pump
        iabp_codes = set(
            [
                "0470041",
                "04700D1",
                "04700Z1",
                "0470341",
                "04703D1",
                "04703Z1",
                "0470441",
                "04704D1",
                "04704Z1",
            ]
        )

        # Impella
        impella_codes = set(["5A0221D"])

        # Ventricular assist
        va_codes = set(["Z95811"])

        for name, code_set in {
            "Intra-aortic Balloon Pump": iabp_codes,
            "Impella": impella_codes,
            "Ventricular Assist": va_codes,
        }.items():
            count = len(
                self.all_df[
                    self.all_df.apply(
                        lambda row: len(set(row["icd_code"]).intersection(code_set))
                        > 0,
                        axis=1,
                    )
                ]
            )

            self._add_table_row(item=name, value=self._pprint_percent(count))

    def _tablegen_mechvent(self) -> None:
        vent_df = pd.read_csv("mimiciv/derived/ventilation.csv")
        # Only mechanical
        vent_df = vent_df[
            vent_df["ventilation_status"].isin(["InvasiveVent", "Tracheostomy"])
        ]

        vent_df = vent_df[vent_df["stay_id"].isin(self.stay_ids)]

        for tc in ["starttime", "endtime"]:
            vent_df[tc] = pd.to_datetime(vent_df[tc])

        def get_duration(row):
            diff_seconds = (row["endtime"] - row["starttime"]).total_seconds()
            return diff_seconds / (60 * 60 * 24)

        vent_df["duration_days"] = vent_df.apply(get_duration, axis=1)

        summed_durations = vent_df.groupby("stay_id")["duration_days"].sum()

        self._add_table_row(
            item="Mean Ventilation length (days)",
            value=self._pprint_mean(summed_durations),
        )

    def populate(self) -> pd.DataFrame:
        tablegen_methods = [m for m in dir(self) if m.startswith("_tablegen")]

        for method_name in tablegen_methods:
            func = getattr(self, method_name)
            print(f"[*] {method_name}")
            func()

        return self.table1


if __name__ == "__main__":
    # sids = pd.read_csv("cache/included_stayids.csv").squeeze("columns")
    sids = pd.read_csv("cache/ecmo_stayids.csv")
    print(f"Total n: {len(sids)}")
    t1generator = Table1Generator(sids["stay_id"].to_list())
    t1 = t1generator.populate()

    print(t1)

    t1.to_csv("results/table1.csv", index=False)
