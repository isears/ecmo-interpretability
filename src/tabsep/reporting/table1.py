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
        @dataclass
        class icd_comorbidity:
            name: str
            codes: List[str]
            points: int

            def __eq__(self, __o: object) -> bool:
                return self.name == __o.name and self.__class__ == __o.__class__

        cci = dict()
        with open("src/tabsep/reporting/cci.json", "r") as f:
            cci = json.load(f)

        comorbidities = [
            icd_comorbidity(
                name=key,
                # Basically, treat all codes as startswith codes
                codes=val["Match Codes"] + val["Startswith Codes"],
                points=int(val["Points"]),
            )
            for key, val in cci.items()
        ]

        def relevant_comorbidities(icd_code_list):
            ret = list()
            for c in comorbidities:
                for code in icd_code_list:
                    if len(list(filter(code.startswith, c.codes))) != 0:
                        ret.append(c)

            return ret

        self.all_df["comorbidities"] = self.all_df["icd_code"].apply(
            relevant_comorbidities
        )

        for c in comorbidities:
            comorbidity_count = len(
                self.all_df[
                    self.all_df["comorbidities"].apply(
                        lambda comorbidity_list: c in comorbidity_list
                    )
                ].index
            )

            self._add_table_row(
                f"[comorbidity] {c.name}", self._pprint_percent(comorbidity_count)
            )

        self.all_df["cci_score"] = self.all_df["comorbidities"].apply(
            lambda comorbidity_list: sum(c.points for c in comorbidity_list)
        )

        self._add_table_row(
            item="[comorbidity] Average CCI",
            value=self._pprint_mean(self.all_df["cci_score"]),
        )

    def populate(self) -> pd.DataFrame:
        tablegen_methods = [m for m in dir(self) if m.startswith("_tablegen")]

        for method_name in tablegen_methods:
            func = getattr(self, method_name)
            print(f"[*] {method_name}")
            func()

        return self.table1


if __name__ == "__main__":
    sids = pd.read_csv("cache/included_stayids.csv").squeeze("columns")
    t1generator = Table1Generator(sids.to_list())
    t1 = t1generator.populate()

    print(t1)

    t1.to_csv("results/table1.csv", index=False)
