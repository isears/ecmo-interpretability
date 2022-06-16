import os
import pandas as pd
from typing import Tuple
import random
import sys
from tabsep import config


def do_random_cut(stay_id: int) -> int:
    # Don't cut within first two timesteps
    ce_df = pd.read_csv(
        f"mimicts/{stay_id}/chartevents_features.csv", nrows=1, index_col="feature_id",
    )
    cutidx = random.randint(2, len(ce_df.columns) - 1)

    return cutidx


def do_sepsis_pos_cut(stay_id: int) -> int:
    sepsis_df = pd.read_csv(
        f"mimicts/{stay_id}/sepsis3_features.csv", index_col="feature_id"
    )
    row = sepsis_df.iloc[0]
    sepsis_idx = int(row[row == 1].index[0])
    # This should be guaranteed by inclusion criteria
    assert sepsis_idx > 1
    # Predict ahead by a certain window
    cutidx = sepsis_idx - config.prediction_timesteps

    assert cutidx >= 2

    return cutidx


def do_sepsis_neg_cut(stay_id: int) -> int:
    sepsis_df = pd.read_csv(
        f"mimicts/{stay_id}/sepsis3_features.csv", index_col="feature_id"
    )
    row = sepsis_df.iloc[0]
    sepsis_idx = int(row[row == 1].index[0])
    # This should be guaranteed by inclusion criteria
    assert sepsis_idx > 1
    # Predict ahead by a certain window
    pos_cutidx = sepsis_idx - config.prediction_timesteps
    neg_cut_limit = pos_cutidx - 1
    assert neg_cut_limit >= 1
    cutidx = random.randint(1, neg_cut_limit)

    return cutidx


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "balanced":
            balanced = True
        elif sys.argv[1] == "unbalanced":
            balanced = False
    else:
        balanced = False

    random.seed(42)
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")
    sepsis3 = pd.read_csv("mimiciv/derived/sepsis3.csv")

    icustays = icustays.merge(sepsis3[["stay_id", "sepsis3"]], how="left", on="stay_id")
    icustays["sepsis3"] = icustays["sepsis3"].fillna(False)

    inclusion_sids = (
        pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()
    )

    icustays = icustays[icustays["stay_id"].isin(inclusion_sids)]

    septic_icustays = icustays[icustays["sepsis3"]]
    nonseptic_icustays = icustays[~icustays["sepsis3"]]

    assert len(septic_icustays) + len(nonseptic_icustays) == len(inclusion_sids)

    print("[*] Postprocessing started with:")
    print(f"\t{len(septic_icustays)} septic stays")
    print(f"\t{len(nonseptic_icustays)} non-septic stays")
    print(f"\t{len(inclusion_sids)} total stays in inclusion criteria")

    if balanced:
        print(
            f"[*] Sampling {len(septic_icustays)} non-septic stays for balanced data..."
        )
        nonseptic_sample = nonseptic_icustays.sample(
            len(septic_icustays), random_state=42
        )
    else:
        nonseptic_sample = nonseptic_icustays

    print(f"[*] Applying random cut to non-septic sample")
    nonseptic_sample["cutidx"] = nonseptic_sample["stay_id"].apply(do_random_cut)
    nonseptic_sample["label"] = 0.0

    print(f"[*] Applying just-before-sepsis cut to septic sample")
    pos_septic_sample = septic_icustays[["stay_id"]]
    pos_septic_sample["cutidx"] = pos_septic_sample["stay_id"].apply(do_sepsis_pos_cut)
    pos_septic_sample["label"] = 1.0

    print(f"[*] Applying random before sepsis cut to septic sample")
    neg_septic_sample = septic_icustays[["stay_id"]]
    neg_septic_sample["cutidx"] = neg_septic_sample["stay_id"].apply(do_sepsis_neg_cut)
    neg_septic_sample["label"] = 0.0

    print(f"[+] Done, saving to disk")
    pd.concat([nonseptic_sample, pos_septic_sample, neg_septic_sample])[
        ["stay_id", "cutidx", "label"]
    ].to_csv("cache/sample_cuts.csv", index=False)
