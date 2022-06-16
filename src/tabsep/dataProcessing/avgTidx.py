"""
Try to figure out when, on average, a feature appears in ICU stays
"""
import pandas as pd
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset, get_feature_labels

# importing the required library
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch

CORES_AVAILABLE = len(os.sched_getaffinity(0))

if __name__ == "__main__":
    feature_of_interest = "Glucose (serum)"
    feature_labels = get_feature_labels()
    feature_idx = feature_labels.index(feature_of_interest)

    cut_sample = pd.read_csv("cache/sample_cuts.csv")

    ds = FileBasedDataset(processed_mimic_path="./mimicts", cut_sample=cut_sample)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=256, num_workers=CORES_AVAILABLE, pin_memory=True,
    )

    for xbatch, _ in dl:
        # X shape will be (batchnum, seq_length, #_features)
        # TODO: get quartiles of ICU stay rather than raw tidx. Need to use padmasks
        x_foi = xbatch[:, :, feature_idx]

        avg_over_sequence = x_foi.mean(axis=0)
        print(avg_over_sequence)
        plt.bar(x=range(len(avg_over_sequence)), height=avg_over_sequence.numpy())
        plt.savefig("bla.png")
        break

