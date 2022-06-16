##########
# Get top 20 features w/maximum attribution at any point during icustay
##########
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep import config


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
    topn = topn.drop(columns="total_importance")

    topn = pd.melt(topn, id_vars="Feature", var_name="Parity")
    ax = sns.barplot(x="Feature", y="value", data=topn, hue="Parity")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title(f"Global Feature Importance ({title})")
    plt.tight_layout()
    plt.savefig(f"results/{title}_importances.png")
    plt.clf()

    return topn


def max_median_importances(att):
    """
    Aggregate importances by finding max value in every time series, than averaging over all time series
    """

    max_attributions = torch.amax(att, dim=1)
    min_attributions = torch.amin(att, dim=1)
    min_mask = (
        torch.max(torch.abs(max_attributions), torch.abs(min_attributions))
        > max_attributions
    )
    max_mask = torch.logical_not(min_mask)

    assert (max_mask.int() + min_mask.int() == 1).all()

    max_absolute_attributions = (
        max_attributions * max_mask.int() + min_attributions * min_mask.int()
    )

    max_absolute_attributions_avg = torch.median(
        max_absolute_attributions, dim=0
    ).values
    importances = pd.DataFrame(
        data={
            "Feature": get_feature_labels(),
            "Median Max Absolute Attribution": max_absolute_attributions_avg.to("cpu")
            .detach()
            .numpy(),
        }
    )

    # Just temporary for topn calculation
    importances["abs"] = importances["Median Max Absolute Attribution"].apply(np.abs)

    topn = importances.nlargest(20, columns="abs")
    topn = topn.drop(columns="abs")

    return topn


if __name__ == "__main__":

    attributions = torch.load(f"{config.model_path}/attributions.pt").detach()
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    X_combined = torch.cat((X_test, X_train), 0).detach()

    print("Loaded data")

    pad_masks = X_combined[:, :, -1]

    titles = ["all", "early", "late"]

    for revised_pad, title in zip(revise_pad(pad_masks), titles):
        att = np.multiply(attributions, np.expand_dims(revised_pad, axis=-1))
        topn = summative_importances(att)

        print(topn)
