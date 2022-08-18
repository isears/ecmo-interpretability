##########
# Get top 20 features w/maximum attribution at any point during icustay
##########
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ecmointerp.dataProcessing.fileBasedDataset import get_feature_labels
from ecmointerp import config


def revise_pad(pm):
    cut_early = np.copy(pm)
    cut_late = np.copy(pm)

    for idx in range(0, pm.shape[0]):
        mask_len = int(pm[idx, :].sum())
        half_mask = mask_len // 2

        cut_early[idx, :] = np.concatenate(
            [np.ones(half_mask), np.zeros(pm.shape[1] - half_mask)]
        )

        cut_late[idx, :] = np.concatenate(
            [
                np.zeros(half_mask),
                np.ones(half_mask),
                np.zeros(pm.shape[1] - (2 * half_mask)),
            ]
        )

    return [pm, cut_early, cut_late]


def summative_importances(att):
    """
    Aggregate importances by summing up all their attributions
    """

    # Separate into negative and positive attrs
    neg_mask = att < 0.0
    neg_attrs = att * neg_mask * -1
    pos_attrs = att * torch.logical_not(neg_mask)

    # Sum over time series dimension, then over batch dimension
    sum_neg_attr = torch.sum(torch.sum(neg_attrs, dim=1), dim=0)
    sum_neg_attr = sum_neg_attr
    sum_pos_attrs = torch.sum(torch.sum(pos_attrs, dim=1), dim=0)

    importances = pd.DataFrame(
        data={
            "Feature": get_feature_labels(),
            "Sum Positive Attributions": sum_pos_attrs.to("cpu").detach().numpy(),
            "Sum Negative Attributions": sum_neg_attr.to("cpu").detach().numpy(),
        }
    )

    importances["total_importance"] = (
        importances["Sum Positive Attributions"]
        + importances["Sum Negative Attributions"]
    )

    topn = importances.nlargest(20, columns="total_importance")
    # topn = topn.drop(columns="total_importance")

    # topn = pd.melt(topn, id_vars="Feature", var_name="Parity")
    # ax = sns.barplot(x="Feature", y="value", data=topn, hue="Parity")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.set_theme()
    ax = sns.barplot(
        x="total_importance", y="Feature", data=topn, orient="h", color="b"
    )
    ax.set_title(f"Global Feature Importance ({title})")
    plt.tight_layout()
    plt.savefig(f"results/{title}_importances.png")
    plt.clf()

    return topn


if __name__ == "__main__":
    # model_dir = "cache/models/singleTst_2022-08-09_21:32:24"
    # title = "base"

    model_dir = "cache/models/transferTst_2022-08-08_20:16:30"
    title = "ecmo"

    attributions = torch.load(f"{model_dir}/attributions.pt").detach()
    X_combined = torch.load(f"{model_dir}/X.pt").detach()
    pad_masks = torch.load(f"{model_dir}/pad_masks.pt")

    print("Loaded data")

    att = np.multiply(attributions, np.expand_dims(pad_masks, axis=-1))
    topn = summative_importances(att)

    print(topn)
