"""
Pre compute metadata so that we don't have to during runtime

For now, just maximum sequence length
"""
import pandas as pd
import json


def compute_max_seq_len() -> int:
    stay_ids = pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()

    max_len = 0
    for sid in stay_ids:
        ce = pd.read_csv(f"mimicts/{sid}/chartevents_features.csv", nrows=1)
        seq_len = len(ce.columns) - 1

        if seq_len > max_len:
            max_len = seq_len

    return max_len


if __name__ == "__main__":
    metadata = dict()

    print("Getting maximum sequence length...")
    metadata["max_len"] = compute_max_seq_len()

    print("Done, saving...")

    with open("cache/metadata.json", "w") as f:
        json.dump(metadata, f)
