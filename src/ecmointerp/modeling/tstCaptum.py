from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    GuidedGradCam,
    GuidedBackprop,
)
import torch
import os
from ecmointerp.modeling.tstImpl import TstOneInput, TSTransformerEncoderClassiregressor
from ecmointerp.modeling.timeseriesCV import TensorBasedDataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ecmointerp import config


torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TensorBasedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, pm=None) -> None:
        self.X = X
        self.y = y
        self.pm = pm

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.pm is not None:
            return self.X[idx], self.y[idx], self.pm[idx]
        else:
            return (
                self.X[idx],
                self.y[idx],
            )


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Detected GPU, using cuda")
        device = "cuda"
    else:
        device = "cpu"

    model_dir = "cache/models/singleTst_2022-08-09_21:32:24"

    # TODO: sync these params up with trainer
    model = TSTransformerEncoderClassiregressor(
        feat_dim=621,
        d_model=128,
        dim_feedforward=256,
        max_len=354,
        n_heads=16,
        num_classes=1,
        num_layers=3,
    ).to(device)

    model.load_state_dict(
        torch.load(
            f"{model_dir}/model.pt",
            map_location=torch.device(device),
        )
    )

    model.eval()

    X_combined = torch.load(f"{model_dir}/X.pt")
    y_combined = torch.load(f"{model_dir}/y.pt")
    pad_masks = torch.load(f"{model_dir}/pad_masks.pt")

    dl = torch.utils.data.DataLoader(
        TensorBasedDataset(X_combined, y_combined, pad_masks),
        batch_size=4,
        num_workers=config.cores_available,
        pin_memory=True,
        drop_last=False,
    )

    attributions_list = list()
    pad_mask_list = list()

    for batch_idx, (xbatch, _, pad_masks) in enumerate(dl):
        with torch.no_grad():  # Computing gradients kills memory
            xbatch = xbatch.to(device)
            pad_masks = pad_masks.to(device)
            xbatch.requires_grad = True

            ig = IntegratedGradients(model)
            attributions = ig.attribute(
                xbatch, additional_forward_args=pad_masks, target=0
            )
            attributions_list.append(attributions.cpu())

            before_mem = torch.cuda.memory_allocated(device) / 2**30
            del attributions
            del pad_masks
            del xbatch
            torch.cuda.empty_cache()
            after_mem = torch.cuda.memory_allocated(device) / 2**30

            print(
                f"batch # {batch_idx} purged memory {before_mem:.4f} -> {after_mem:.4f}"
            )

    attributions_all = torch.concat(attributions_list, dim=0)
    print(f"Saving attributions to {model_dir}/attributions.pt")
    torch.save(attributions_all, f"{model_dir}/attributions.pt")
