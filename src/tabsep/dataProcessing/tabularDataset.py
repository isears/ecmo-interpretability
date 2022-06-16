import os.path
import torch
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from typing import List
import json


class TabularDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        processed_mimic_path: str,
        stay_ids: List[int] = None,
        feature_ids: List[int] = None,
        labels: pd.DataFrame = None,
    ):

        print(f"[{type(self).__name__}] Initializing dataset...")
        if stay_ids is None:
            self.stay_ids = (
                pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()
            )
        else:
            self.stay_ids = stay_ids

        if feature_ids is None:
            self.feature_ids = (
                pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
            )
        else:
            self.feature_ids = feature_ids

        if labels is None:
            self.labels = pd.read_csv("cache/labels.csv", index_col=0)
            self.labels = self.labels.reindex(self.stay_ids)
        else:
            self.labels = labels

        assert not self.labels["label"].isna().any(), "[-] Labels had nan values"
        assert len(self.labels) == len(
            self.stay_ids
        ), "[-] Mismatch between stay ids and labels"

        print(f"\tStay IDs: {len(self.stay_ids)}")
        print(f"\tFeatures: {len(self.feature_ids)}")

        self.processed_mimic_path = processed_mimic_path

        print(f"[{type(self).__name__}] Dataset initialization complete")

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        chartevents = pd.read_csv(
            f"cache/simpledp/{stay_id}/chartevents.csv",
            dtype={"stay_id": "int", "valuenum": "float", "itemid": "int"},
            usecols=["stay_id", "itemid", "valuenum"],
        )
        chartevents = (
            chartevents[["itemid", "valuenum"]].groupby("itemid").apply("mean")
        )

        chartevents = chartevents.reindex(self.feature_ids)
        chartevents = chartevents.fillna(0.0)

        return chartevents.values[:, 0], self.labels.loc[stay_id].values[0]


if __name__ == "__main__":
    ds = TabularDataset("./cache/simpledp")

    dl = torch.utils.data.DataLoader(ds, num_workers=2, batch_size=4, pin_memory=True)

    print("Printing first few batches:")
    for batchnum, (X, Y) in enumerate(dl):
        print(f"Batch number: {batchnum}")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(X)
        print(Y)

        if batchnum == 5:
            break
