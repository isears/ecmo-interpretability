"""
Fine-tune a pretrained TST on ECMO data only
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ecmointerp.modeling.timeseriesCV import TstWrapper, load_to_mem
import pandas as pd
from ecmointerp.dataProcessing.fileBasedDataset import FileBasedDataset
from ecmointerp.modeling.tstImpl import TSTransformerEncoderClassiregressor
from ecmointerp import config
import os
import datetime


if __name__ == "__main__":
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    model = TSTransformerEncoderClassiregressor(
        feat_dim=621,
        d_model=128,
        dim_feedforward=256,
        max_len=354,
        n_heads=16,
        num_classes=1,
        num_layers=3,
    ).to("cuda")

    model.load_state_dict(
        torch.load(
            f"{config.model_path}/model.pt",
            map_location=torch.device("cuda"),
        )
    )

    # Freeze all but last layer
    # model_layers = [(name, param) for name, param in model.named_parameters()]
    # for idx, (name, param) in enumerate(model_layers):
    #     if not idx > len(model_layers) - 3:
    #         print(f"Freezing layer: {name}")
    #         param.requires_grad = False
    #     else:
    #         print(f"Leaving layer {name} unfrozen")

    # print(
    #     f"Unfrozen layers: {[name for name, param in model.named_parameters() if param.requires_grad]}"
    # )

    tst = TstWrapper(max_epochs=2)
    all_sids = pd.read_csv("cache/ecmo_stayids.csv").squeeze("columns").to_list()

    ds = FileBasedDataset(processed_mimic_path="./mimicts", stay_ids=all_sids)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=tst.batch_size,
        num_workers=config.cores_available,
        pin_memory=True,
    )

    print("[+] Data loaded, training...")
    tst.train(train_dl=dl, model=model)

    save_path = f"cache/models/transferTst_{start_time_str}"
    print(f"[+] Training complete, saving to {save_path}")
    X, y, pm = load_to_mem(dl)
    preds = tst.decision_function(X, pm)
    score = roc_auc_score(y, preds)
    print(f"Validation score: {score}")

    os.mkdir(save_path)
    torch.save(tst.model.state_dict(), f"{save_path}/model.pt")

    torch.save(preds, f"{save_path}/preds.pt")
    torch.save(X, f"{save_path}/X.pt")
    torch.save(X, f"{save_path}/y.pt")
    torch.save(pm, f"{save_path}/pad_masks.pt")

    with open(f"{save_path}/roc_auc_score.txt", "w") as f:
        f.write(str(score))
        f.write("\n")
