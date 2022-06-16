"""
Train single transformer model for downstream analysis
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabsep.modeling.timeseriesCV import TstWrapper, load_to_mem
import pandas as pd
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
import os
import datetime

CORES_AVAILABLE = len(os.sched_getaffinity(0))


if __name__ == "__main__":
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    print("[*] Loading data to memory")
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    cut_sample = cut_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    ds = FileBasedDataset(processed_mimic_path="./mimicts", cut_sample=cut_sample)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=256, num_workers=CORES_AVAILABLE, pin_memory=True,
    )

    X, y = load_to_mem(dl)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print("[+] Data loaded, training...")
    tst = TstWrapper(max_epochs=100)  # To allow early stopper to do its thing
    tst.fit(X_train, y_train, use_es=True, X_valid=X_test, y_valid=y_test)

    save_path = f"cache/models/singleTst_{start_time_str}"
    print(f"[+] Training complete, saving to {save_path}")

    os.mkdir(save_path)
    torch.save(tst.model.state_dict(), f"{save_path}/model.pt")
    torch.save(X_test, f"{save_path}/X_test.pt")
    torch.save(y_test, f"{save_path}/y_test.pt")
    torch.save(X_train, f"{save_path}/X_train.pt")
    torch.save(y_train, f"{save_path}/y_train.pt")

    preds = tst.decision_function(X_test)
    torch.save(preds, f"{save_path}/preds.pt")
    score = roc_auc_score(y_test, preds)
    print(f"Validation score: {score}")

    with open(f"{save_path}/roc_auc_score.txt", "w") as f:
        f.write(str(score))
        f.write("\n")
