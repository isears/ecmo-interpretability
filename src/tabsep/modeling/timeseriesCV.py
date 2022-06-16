import pickle
from dataclasses import dataclass
import sys
from typing import Callable
import torch
import numpy as np
import pandas as pd
import scipy.stats as st
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tabsep.modeling.tstImpl import TSTransformerEncoderClassiregressor, AdamW
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import os
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, make_scorer


CORES_AVAILABLE = len(os.sched_getaffinity(0))
torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@dataclass
class SingleCVResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


class CVResults:
    @classmethod
    def load(cls, filename) -> "CVResults":
        with open(filename, "rb") as f:
            return pickle.load(f)

    def __init__(self, clf_name) -> None:
        self.results = list()
        self.clf_name = clf_name

    def add_result(self, y_true, y_score) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        self.results.append(SingleCVResult(fpr, tpr, thresholds, auc))

        # For compatibility w/sklearn scorers
        return auc

    def get_scorer(self) -> Callable:
        metric = lambda y_t, y_s: self.add_result(y_t, y_s)
        return make_scorer(metric, needs_proba=True)

    def print_report(self) -> None:
        aucs = np.array([res.auc for res in self.results])
        print(f"All scores: {aucs}")
        print(f"Score mean: {aucs.mean()}")
        print(f"Score std: {aucs.std()}")
        print(
            f"95% CI: {st.t.interval(alpha=0.95, df=len(aucs)-1, loc=aucs.mean(), scale=st.sem(aucs))}"
        )

        with open(f"results/{self.clf_name}.cvresult", "wb") as f:
            pickle.dump(self, f)


class FeatureScaler(StandardScaler):
    """
    Scale the features one at a time

    Assumes shape of data is (n_samples, seq_len, n_features)
    """

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None, sample_weight=None):
        self.feature_scalers = list()

        for feature_idx in range(0, X.shape[-1]):  # Assuming feature_dim is last dim
            feature_scaler = StandardScaler(
                copy=self.copy, with_mean=self.with_mean, with_std=self.with_std
            )
            feature_scaler.fit(X[:, :, feature_idx])
            self.feature_scalers.append(feature_scaler)

        return self

    def transform(self, X, copy=None):
        return np.stack(
            [f.transform(X[:, :, idx]) for idx, f in enumerate(self.feature_scalers)],
            axis=-1,
        )

    def inverse_transform(self, X, copy=None):
        return np.stack(
            [
                f.reverse_tranform(X[:, :, idx])
                for idx, f in enumerate(self.feature_scalers)
            ],
            axis=1,
        )


class Ts2TabTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.reshape(X.shape[0], (X.shape[1] * X.shape[2]))


class TensorBasedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, padding_masks) -> None:
        self.X = X
        self.y = y
        self.padding_masks = padding_masks

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            self.padding_masks[idx],
        )


class TstWrapper(BaseEstimator, ClassifierMixin):
    # TODO: max_len must be set dynamically based on cache metadata
    def __init__(
        # From TST paper: hyperparameters that perform generally well
        self,
        # Fit params
        max_epochs=7,  # This is not specified by paper, depends on dataset size
        batch_size=128,  # Should be 128, but gpu can't handle it
        optimizer_cls=AdamW,
        # TST params
        d_model=128,
        dim_feedforward=256,
        max_len=120,
        n_heads=16,
        num_classes=1,
        num_layers=3,
    ) -> None:

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.n_heads = n_heads
        self.num_classes = num_classes
        self.num_layers = num_layers

    @staticmethod
    def _unwrap_X_padmask(X_packaged):
        # For compatibility with scikit
        X, padding_masks = (
            X_packaged[:, :, 0:-1],
            X_packaged[:, :, -1] == 1,
        )

        return X, padding_masks

    def fit(self, X, y, use_es=False, X_valid=None, y_valid=None):
        # scikit boilerplate
        self.classes_ = np.array([0.0, 1.0])
        # original_y_shape = y.shape
        # self.classes_, y = np.unique(y, return_inverse=True)
        # y = torch.reshape(torch.tensor(y), original_y_shape).float()  # re-torch y

        model = TSTransformerEncoderClassiregressor(
            feat_dim=X.shape[-1] - 1,  # -1 for padding mask
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            max_len=self.max_len,
            n_heads=self.n_heads,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
        ).to("cuda")

        optimizer = self.optimizer_cls(model.parameters())
        criterion = torch.nn.BCELoss()
        # TODO: eventually may have to do two types of early stopping implementations:
        # One "fair" early stopping for comparison w/LR
        # One "optimistic" early stopping for single fold model building
        # Current impl is optimistic but does not run under CV
        es = EarlyStopping(patience=5)

        X_unpacked, padding_masks = TstWrapper._unwrap_X_padmask(X)

        gpuLoader = torch.utils.data.DataLoader(
            TensorBasedDataset(X_unpacked, y, padding_masks),
            batch_size=self.batch_size,
            num_workers=CORES_AVAILABLE,
            pin_memory=True,
            drop_last=False,
        )

        if use_es:
            assert X_valid is not None and y_valid is not None
            X_valid_unpacked, pm_valid = TstWrapper._unwrap_X_padmask(X_valid)

            validGpuLoader = torch.utils.data.DataLoader(
                TensorBasedDataset(X_valid_unpacked, y_valid, pm_valid),
                batch_size=self.batch_size,
                num_workers=CORES_AVAILABLE,
                pin_memory=True,
                drop_last=False,
            )

        for epoch_idx in range(0, self.max_epochs):
            model.train()
            optimizer.zero_grad()

            for batch_idx, (batch_X, batch_y, batch_padding_masks) in enumerate(
                gpuLoader
            ):
                outputs = model.forward(
                    batch_X.to("cuda"), batch_padding_masks.to("cuda")
                )
                loss = criterion(outputs, torch.unsqueeze(batch_y, 1).to("cuda"))
                loss.backward()
                # TODO: what's the significance of this?
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()

            # Do early stopping
            if use_es:
                model.eval()
                y_pred = torch.Tensor().to("cuda")
                y_actual = torch.Tensor().to("cuda")

                for bXv, byv, pmv in validGpuLoader:
                    this_y_pred = model(bXv.to("cuda"), pmv.to("cuda"))
                    y_pred = torch.cat((y_pred, this_y_pred))
                    y_actual = torch.cat((y_actual, byv.to("cuda")))

                validation_loss = log_loss(
                    y_actual.detach().to("cpu"), y_pred.detach().to("cpu")
                )

                es(validation_loss, model.state_dict())

                if es.early_stop:
                    print(f"Stopped training @ epoch {epoch_idx}")
                    break

        if es.saved_best_weights:
            model.load_state_dict(es.saved_best_weights)

        self.model = model

        return self

    # This seems to be the function used by scikit cv_loop, which is all we really care about right now
    def decision_function(self, X):
        with torch.no_grad():
            # TODO: assuming validation set small enough to fit into gpu mem w/out batching?
            # Also, can the TST model handle a new shape?
            X_unpacked, padding_masks = TstWrapper._unwrap_X_padmask(X)
            y_pred = self.model(X_unpacked.to("cuda"), padding_masks.to("cuda"))

            # send model to cpu at end so that it's not taking up GPU space
            print("[*] Fold done, sending model to CPU")
            self.model.to("cpu")

            return torch.squeeze(y_pred).to("cpu")  # sklearn needs to do cpu ops

    def predict(self, X):
        return self.decision_function(X)


def load_to_mem(dl: torch.utils.data.DataLoader):
    """
    Necessary for scikit models to have everything in memory
    """
    print("[*] Loading all data from disk")
    X_all, y_all = [torch.tensor([])] * 2
    for X, y in dl:
        X_all = torch.cat((X_all, X), dim=0)
        y_all = torch.cat((y_all, y), dim=0)

    print("[+] Data loading complete:")
    print(f"\tX shape: {X_all.shape}")
    print(f"\ty shape: {y_all.shape}")
    return X_all, torch.squeeze(y_all)


def doCV(clf, n_jobs=-1):
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    cut_sample = cut_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    ds = FileBasedDataset(processed_mimic_path="./mimicts", cut_sample=cut_sample)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=256, num_workers=CORES_AVAILABLE, pin_memory=True,
    )

    X, y = load_to_mem(dl)

    cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_res = CVResults(clf.__class__.__name__)

    # TODO: CVResults gets copied when n_jobs > 1, breaks everything
    cross_val_score(
        clf,
        X,
        y,
        cv=cv_splitter,
        scoring=cv_res.get_scorer(),
        n_jobs=n_jobs,
        verbose=1,
    )

    cv_res.print_report()

    return cv_res


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "LR"

    models = {
        "LR Scaled": make_pipeline(
            FeatureScaler(), Ts2TabTransformer(), LogisticRegression(max_iter=1e7)
        ),
        "LR": make_pipeline(Ts2TabTransformer(), LogisticRegression()),
        "XGBOOST": XGBClassifier(),
        "TST": TstWrapper(),
    }

    job_config = {"LR": 1, "TST": 1, "XGBOOST": 11}

    if model_name == "all":

        for idx, (model_name, clf) in enumerate(models.items()):
            doCV(clf, job_config[model_name])

    elif model_name in models:
        clf = models[model_name]
        doCV(clf, job_config[model_name])
    else:
        raise ValueError(f"No model named {sys.argv[1]}. Pick from {models.keys()}")
