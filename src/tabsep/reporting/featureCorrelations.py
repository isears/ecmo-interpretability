import torch
from tabsep import config
import numpy as np
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import seaborn as sns


class FeatureCorrelator:
    def __init__(self, feature: str) -> None:
        self.feature = feature

        self.attributions = (
            torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
        )

        X_test = torch.load(f"{config.model_path}/X_test.pt")
        X_train = torch.load(f"{config.model_path}/X_train.pt")
        X = torch.cat((X_test, X_train), 0).detach().numpy()
        self.pad_masks = X[:, :, -1]

        self.feature_labels = get_feature_labels()

        if self.feature not in self.feature_labels:
            raise ValueError(f"{self.feature} not found, check format")

        self.feature_idx = self.feature_labels.index(feature)
        self.correlation_attribs = None

    def _std_dev_clip(self, x: np.ndarray, n_stdevs: int):
        std_dev = x[:, self.feature_idx].std()
        ret = x[
            ~np.isclose(
                correlation_attribs[:, self.feature_idx], 0.0, atol=std_dev * n_stdevs
            )
        ]

        return ret

    def direct_comparison(self):
        """
        Matches feature attribs with other features only at that specific point in time

        Drawback: if feature x correlates w/feature y, 
        but only some time before feature y is recorded, the correlation will be missed 
        """
        saved_attribs = list()

        for idx in range(0, self.attributions.shape[0]):  # Iterate over all icustays
            this_stay_attributions = self.attributions[idx]
            this_stay_length = int(self.pad_masks[idx].sum())
            clipped_attributions = this_stay_attributions[0:this_stay_length]

            # Here we only look at time points where the feature of interest is non-zero attributable
            # TODO: do we want to go further, and add an attribution cut-off?
            original_feature_clipped_attr = clipped_attributions[:, self.feature_idx]
            non_zero_attr_indices = [
                jdx
                for jdx in range(0, len(original_feature_clipped_attr))
                if not np.isclose(original_feature_clipped_attr[jdx], 0.0)
            ]

            clipped_attributions = clipped_attributions[non_zero_attr_indices, :]

            saved_attribs.append(clipped_attributions)

        # Should be of the shape (# icu stays * avg. icu stay length, # features)
        correlation_attribs = np.concatenate(saved_attribs)
        self.correlation_attribs = correlation_attribs
        assert correlation_attribs.shape[1] == self.attributions.shape[-1]

        return correlation_attribs

    def absolute_max_comparison(self):
        """
        Matches max of absolute feature attribution with other features' max of absolute attributions
        """
        maximums = np.max(self.attributions, axis=1)
        minimums = np.min(self.attributions, axis=1)

        # Select by highest absolute value
        max_mask = maximums > np.abs(minimums)
        min_mask = np.logical_not(max_mask)
        correlation_attribs = maximums * max_mask + minimums * min_mask

        # Drop anything within one standard deviation of 0.0
        before_size = correlation_attribs.shape[0]
        std_dev = correlation_attribs[:, self.feature_idx].std()
        correlation_attribs = correlation_attribs[
            ~np.isclose(correlation_attribs[:, self.feature_idx], 0.0, atol=std_dev * 1)
        ]
        after_size = correlation_attribs.shape[0]

        print(f"[*] Filtered {before_size - after_size}, final size {after_size}")

        return correlation_attribs


if __name__ == "__main__":
    fc = FeatureCorrelator("Foley")

    correlation_attribs = fc.absolute_max_comparison()

    rvals = list()
    yvals = list()
    slopes = list()
    intercepts = list()
    feature_indices = list()
    feature_names = list()

    for idx in range(0, correlation_attribs.shape[1]):
        if idx == fc.feature_idx:  # Correlating feature w/itself will obviously be best
            continue
        x = correlation_attribs[:, fc.feature_idx]
        y = correlation_attribs[:, idx]

        try:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        except ValueError:  # If we filter out all examples by mistake
            slope, intercept, r_value, std_err = [0.0] * 4
            p_value = 1.0

        rvals.append(r_value)
        yvals.append(y)
        feature_indices.append(idx)
        feature_names.append(fc.feature_labels[idx])
        slopes.append(slope)

    df = pd.DataFrame(
        {"Feature": feature_names, "Rval": rvals, "Slope": slopes, "Yvals": yvals},
        index=feature_indices,
    )

    topn = df.nlargest(columns="Rval", n=10)

    print(topn)

    plt.figure()
    ax = sns.barplot(x="Feature", y="Rval", data=topn, color="blue")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title(f"Top features correlating with: {fc.feature}")

    plt.tight_layout()
    plt.savefig(f"results/correlations.png")
    plt.clf()

    plt.figure()
    ax = sns.regplot(x=x, y=topn.iloc[0]["Yvals"])
    ax.set_title(
        f"Attribution Correlation between {fc.feature} and\n{topn.iloc[0]['Feature']}"
    )
    ax.set_xlabel(fc.feature)
    ax.set_ylabel(topn.iloc[0]["Feature"])
    plt.tight_layout()
    plt.savefig(f"results/top_correlation.png")
