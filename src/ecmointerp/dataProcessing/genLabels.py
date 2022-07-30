import pandas as pd
import os

if __name__ == "__main__":
    stay_ids = [int(dirname) for dirname in os.listdir("cache/mimicts")]

    labels = pd.DataFrame(index=stay_ids, data={"label": [0] * len(stay_ids)})

    def get_sepsis(stay_id: int):
        df = pd.read_csv(f"cache/mimicts/{stay_id}/sepsis3_features.csv")
        # If there's an entry in the dataframe, there must be sepsis
        # Super simplistic for now
        return int(len(df) != 0)

    labels["label"] = labels.index.map(get_sepsis)

    print(
        f"Generated labels, (+) label prevalance: {labels['label'].sum() / len(labels):.4f}"
    )

    labels.to_csv("cache/labels.csv")
