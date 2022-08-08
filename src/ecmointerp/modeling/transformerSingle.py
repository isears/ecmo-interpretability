"""
Train single transformer model for downstream analysis
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ecmointerp.modeling.timeseriesCV import TstWrapper, load_to_mem
import pandas as pd
from ecmointerp.dataProcessing.fileBasedDataset import FileBasedDataset
from ecmointerp import config
import os
import datetime


if __name__ == "__main__":
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tst = TstWrapper(max_epochs=100)
    all_sids = pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()
    train_sids, test_sids = train_test_split(all_sids, test_size=0.1, random_state=42)

    train_ds = FileBasedDataset(processed_mimic_path="./mimicts", stay_ids=train_sids)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=tst.batch_size,
        num_workers=config.cores_available,
        pin_memory=True,
    )

    test_ds = FileBasedDataset(processed_mimic_path="./mimicts", stay_ids=test_sids)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=tst.batch_size,
        num_workers=config.cores_available,
        pin_memory=True,
    )

    print("[+] Data loaded, training...")
    tst.train(train_dl=train_dl, use_es=True, valid_dl=test_dl)

    save_path = f"cache/models/singleTst_{start_time_str}"
    print(f"[+] Training complete, saving to {save_path}")
    X, y, pm = load_to_mem(test_dl)
    preds = tst.decision_function(X, pm)
    score = roc_auc_score(y, preds)
    print(f"Validation score: {score}")

    os.mkdir(save_path)
    torch.save(tst.model.state_dict(), f"{save_path}/model.pt")
    pd.Series(name="stay_id", data=train_sids).to_csv(f"{save_path}/train_sids.csv")
    pd.Series(name="stay_id", data=test_sids).to_csv(f"{save_path}/test_sids.csv")

    torch.save(preds, f"{save_path}/preds.pt")

    with open(f"{save_path}/roc_auc_score.txt", "w") as f:
        f.write(str(score))
        f.write("\n")
