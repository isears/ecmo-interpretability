import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep import config


if __name__ == "__main__":
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    y_test = torch.load(f"{config.model_path}/y_test.pt")
    preds = torch.load(f"{config.model_path}/preds.pt")
    attributions = torch.load(f"{config.model_path}/attributions.pt")

    pad_masks = X_test[:, :, -1] == 1
    # X_test = X_test[:, :, :-1]

    sample_idx = np.argmax(
        torch.logical_and(
            torch.logical_and(y_test == 1, torch.sum(X_test[:, :, -1], dim=1) > 3,),
            torch.logical_and(torch.sum(X_test[:, :, -1], dim=1) < 40, preds > 0.65,),
        )
    )

    print(f"Analyzing local importance of idx {sample_idx}")
    sample_case = pd.DataFrame(attributions[sample_idx].cpu().detach().numpy())

    sample_case.columns = get_feature_labels()
    # Truncate by padding mask
    sample_case = sample_case[pad_masks[sample_idx].tolist()]
    max_absolute_attribution = sample_case.abs().apply(lambda col: col.sum())
    top_n_features = max_absolute_attribution.nlargest(n=10).index

    sample_case = sample_case.drop(
        columns=[c for c in sample_case.columns if c not in top_n_features]
    )

    sample_case.index = sample_case.index * (config.timestep_seconds / (60 * 60))

    sample_case.index.name = "Time in ICU (hrs.)"

    ax = sns.heatmap(sample_case, linewidths=0.01, linecolor="black")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(
        f"Validation set idx {sample_idx} prediction {preds[sample_idx]:.2f} actual {y_test[sample_idx]:.2f}"
    )
    plt.savefig("results/local_importance.png", bbox_inches="tight")
