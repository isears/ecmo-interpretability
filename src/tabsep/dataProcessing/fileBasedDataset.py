import os.path
import torch
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from typing import List
import json


def get_feature_labels():
    """
    Returns feature labels in-order of their appearance in X
    """
    feature_ids = (
        pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
    )
    d_items = pd.read_csv("mimiciv/icu/d_items.csv", index_col="itemid")
    d_items = d_items.reindex(feature_ids)

    assert len(d_items) == len(feature_ids)

    return d_items["label"].to_list()


class FileBasedDataset(torch.utils.data.Dataset):
    def __init__(self, processed_mimic_path: str, cut_sample: pd.DataFrame):

        print(f"[{type(self).__name__}] Initializing dataset...")

        self.feature_ids = (
            pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
        )

        self.cut_sample = cut_sample

        print(f"\tExamples: {len(self.cut_sample)}")
        print(f"\tFeatures: {len(self.feature_ids)}")

        self.processed_mimic_path = processed_mimic_path

        try:
            with open("cache/metadata.json", "r") as f:
                self.max_len = int(json.load(f)["max_len"])
        except FileNotFoundError:
            print(
                f"[{type(self).__name__}] Failed to load metadata. Computing maximum length, this may take some time..."
            )
            self.max_len = 0
            for sid in self.cut_sample["stay_id"].to_list():
                ce = pd.read_csv(
                    f"{processed_mimic_path}/{sid}/chartevents_features.csv", nrows=1
                )
                seq_len = len(ce.columns) - 1

                if seq_len > self.max_len:
                    self.max_len = seq_len

        print(f"\tMax length: {self.max_len}")

        print(f"[{type(self).__name__}] Dataset initialization complete")

    def maxlen_padmask_collate(self, batch):
        for idx, (X, y) in enumerate(batch):
            actual_len = X.shape[1]

            assert actual_len < self.max_len

            pad_mask = torch.ones(actual_len)
            X_mod = pad(X, (self.max_len - actual_len, 0), mode="constant", value=0.0)

            pad_mask = pad(
                pad_mask, (self.max_len - actual_len, 0), mode="constant", value=0.0
            )

            batch[idx] = (X_mod.T, y, pad_mask)

        X = torch.stack([X for X, _, _ in batch], dim=0)
        y = torch.stack([Y for _, Y, _ in batch], dim=0)
        pad_mask = torch.stack([pad_mask for _, _, pad_mask in batch], dim=0)

        raise NotImplemented
        return X.float(), y.float(), pad_mask.int()

    def maxlen_padmask_collate_nopadret(self, batch):
        """
        For compatibility with scikit learn
        * If using this method, remember to remove the padmask from X in model
        """
        X, y, pad_mask = self.maxlen_padmask_collate(batch)
        X_and_pad = torch.cat((X, torch.unsqueeze(pad_mask, dim=-1)), dim=-1)
        return X_and_pad, y

    def __len__(self):
        return len(self.cut_sample)

    def __getitem__(self, index: int):
        stay_id = self.cut_sample["stay_id"].iloc[index]
        Y = torch.tensor(self.cut_sample["label"].iloc[index])
        Y = torch.unsqueeze(Y, 0)
        cutidx = self.cut_sample["cutidx"].iloc[index]

        # Features
        # Ensures every example has a sequence length of at least 1
        combined_features = pd.DataFrame(columns=["feature_id", "0"])

        for feature_file in [
            "chartevents_features.csv",
            "outputevents_features.csv",
            "inputevent_features.csv",
        ]:
            full_path = f"{self.processed_mimic_path}/{stay_id}/{feature_file}"

            if os.path.exists(full_path):
                curr_features = pd.read_csv(
                    full_path,
                    usecols=list(range(0, cutidx + 1)),
                    index_col="feature_id",
                )

                combined_features = pd.concat([combined_features, curr_features])

        # Make sure all itemids are represented in order, add 0-tensors where missing
        combined_features = combined_features.reindex(
            self.feature_ids
        )  # Need to add any itemids that are missing
        # TODO: could probably do imputation better (but maybe during preprocessing)
        combined_features = combined_features.fillna(0.0)
        X = torch.tensor(combined_features.values)

        # Pad to maxlen
        actual_len = X.shape[1]
        assert actual_len < self.max_len
        pad_mask = torch.ones(actual_len)
        # TODO: transform here b/c the way TST expects it isn't the typical convention
        X_mod = pad(X, (0, self.max_len - actual_len), mode="constant", value=0.0).T
        pad_mask = pad(
            pad_mask, (0, self.max_len - actual_len), mode="constant", value=0.0
        )

        # Put pad as last "feature" in X for compatibility w/scikit
        X_and_pad = torch.cat((X_mod, torch.unsqueeze(pad_mask, dim=-1)), dim=-1)

        return X_and_pad.float(), Y.float()


def demo(dl):
    print("Printing first few batches:")
    for batchnum, (X, Y, pad_mask) in enumerate(dl):
        print(f"Batch number: {batchnum}")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(X)
        print(Y)

        if batchnum == 5:
            break


def get_label_prevalence(dl):
    y_tot = torch.tensor(0.0)
    for batchnum, (X, Y, pad_mask) in enumerate(dl):
        y_tot += torch.sum(Y)

    print(f"Postivie Ys: {y_tot / (batchnum * dl.batch_size)}")


if __name__ == "__main__":
    sample_cuts = pd.read_csv("cache/sample_cuts.csv")
    ds = FileBasedDataset("./mimicts", cut_sample=sample_cuts)

    dl = torch.utils.data.DataLoader(
        ds,
        collate_fn=ds.maxlen_padmask_collate,
        num_workers=16,
        batch_size=4,
        pin_memory=True,
    )

    get_label_prevalence(dl)
