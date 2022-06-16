from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    GuidedGradCam,
    GuidedBackprop,
)
import torch
import os
from tabsep.modeling.tstImpl import TstOneInput, TSTransformerEncoderClassiregressor
from tabsep.modeling.timeseriesCV import TensorBasedDataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabsep import config


torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


CORES_AVAILABLE = len(os.sched_getaffinity(0))


class TensorBasedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
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

    # TODO: sync these params up with trainer
    model = TSTransformerEncoderClassiregressor(
        feat_dim=621,
        d_model=128,
        dim_feedforward=256,
        max_len=120,
        n_heads=16,
        num_classes=1,
        num_layers=3,
    ).to(device)

    model.load_state_dict(
        torch.load(f"{config.model_path}/model.pt", map_location=torch.device(device),)
    )

    model.eval()

    X_test = torch.load(f"{config.model_path}/X_test.pt")
    y_test = torch.load(f"{config.model_path}/y_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    y_train = torch.load(f"{config.model_path}/y_train.pt")

    X_combined = torch.cat((X_test, X_train), 0)
    y_combined = torch.cat((y_test, y_train), 0)

    dl = torch.utils.data.DataLoader(
        TensorBasedDataset(X_combined, y_combined),
        batch_size=32,
        num_workers=CORES_AVAILABLE,
        pin_memory=True,
        drop_last=False,
    )

    attributions_list = list()
    pad_mask_list = list()

    for batch_idx, (xbatch, _) in enumerate(dl):
        with torch.no_grad():  # Computing gradients kills memory
            xbatch = xbatch.to(device)

            pad_masks = xbatch[:, :, -1] == 1
            xbatch = xbatch[:, :, :-1]

            xbatch.requires_grad = True

            ig = IntegratedGradients(model)
            attributions = ig.attribute(
                xbatch, additional_forward_args=pad_masks, target=0
            )
            attributions_list.append(attributions.cpu())

            before_mem = torch.cuda.memory_allocated(device) / 2 ** 30
            del attributions
            del pad_masks
            del xbatch
            torch.cuda.empty_cache()
            after_mem = torch.cuda.memory_allocated(device) / 2 ** 30

            print(
                f"batch # {batch_idx} purged memory {before_mem:.4f} -> {after_mem:.4f}"
            )

    attributions_all = torch.concat(attributions_list, dim=0)
    print(f"Saving attributions to {config.model_path}/attributions.pt")
    torch.save(attributions_all, f"{config.model_path}/attributions.pt")
