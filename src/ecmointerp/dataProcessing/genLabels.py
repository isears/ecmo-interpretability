import pandas as pd
import os


def _handle_labels(hadm_group, label_codes):
    ret = pd.Series(index=label_codes.keys(), dtype="float")
    for outcome, relevant_codes in label_codes.items():
        ret.loc[outcome] = hadm_group["icd_code"].isin(relevant_codes).any()

    return ret


def do_agg(stay_ids):
    label_codes = dict()
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")

    icustays = icustays[icustays["stay_id"].isin(stay_ids)]

    # Labels
    diagnoses = pd.read_csv("mimiciv/hosp/diagnoses_icd.csv")
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(icustays["hadm_id"])]

    # Thrombosis
    dvt_codes = (
        diagnoses[diagnoses["icd_code"].str.startswith("I8240")]["icd_code"]
        .unique()
        .tolist()
    )
    dvt_codes += (
        diagnoses[diagnoses["icd_code"].str.startswith("4534")]["icd_code"]
        .unique()
        .tolist()
    )

    pe_codes = (
        diagnoses[diagnoses["icd_code"].str.startswith("I26")]["icd_code"]
        .unique()
        .tolist()
    )
    pe_codes += (
        diagnoses[diagnoses["icd_code"].str.startswith("4151")]["icd_code"]
        .unique()
        .tolist()
    )
    oxygenator_thrombosis_codes = (
        diagnoses[diagnoses["icd_code"].isin(["T82867A", "99672"])]["icd_code"]
        .unique()
        .tolist()
    )
    unspecified_thrombosis_codes = (
        diagnoses[
            (diagnoses["icd_code"] == "4449")
            | (diagnoses["icd_code"].str.startswith("I74"))
        ]["icd_code"]
        .unique()
        .tolist()
    )

    label_codes["Thrombosis"] = (
        dvt_codes
        + pe_codes
        + oxygenator_thrombosis_codes
        + unspecified_thrombosis_codes
    )

    # Hemorrhage
    ic_hemorrhage_codes = (
        diagnoses[diagnoses["icd_code"] == "431"]["icd_code"].unique().tolist()
    )
    ic_hemorrhage_codes += (
        diagnoses[diagnoses["icd_code"].str.startswith("I61")]["icd_code"]
        .unique()
        .tolist()
    )
    gi_hemorrhage_codes = ["578", "5780", "5781", "5789", "K922"]
    csite_hemorrhage_codes = ["99674", "T82838A"]
    ssite_hemorrhage_codes = ["L7622", "99811"]
    pulm_hemorrhage_codes = ["R0489", "78639"]
    epistaxis_codes = ["R040", "7847"]
    nonspecific_bleeding_codes = ["99811", "I97418", "I9742", "I97618", "I9762"]
    label_codes["Hemorrhage"] = (
        ic_hemorrhage_codes
        + gi_hemorrhage_codes
        + csite_hemorrhage_codes
        + ssite_hemorrhage_codes
        + pulm_hemorrhage_codes
        + epistaxis_codes
        + nonspecific_bleeding_codes
    )

    labels_df = diagnoses.groupby("hadm_id").apply(
        _handle_labels, label_codes=label_codes
    )
    labels_df = icustays.merge(
        labels_df, how="left", left_on="hadm_id", right_index=True
    )

    labels_df = labels_df[["stay_id"] + list(label_codes.keys())]
    labels_df = labels_df.fillna(False)

    return labels_df


if __name__ == "__main__":
    stay_ids = pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()

    labels_df = do_agg(stay_ids)

    labels_df.to_csv("cache/labels.csv", index=False)
