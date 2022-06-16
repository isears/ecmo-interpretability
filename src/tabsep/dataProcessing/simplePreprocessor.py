import dask.dataframe as dd
import pandas as pd
import os

all_inclusive_dtypes = {
    # Chartevents
    "subject_id": "int",
    "hadm_id": "int",
    "stay_id": "int",
    "charttime": "object",
    "storetime": "object",
    "itemid": "int",
    "value": "object",
    "valueuom": "object",
    "warning": "object",
    "valuenum": "float",
    # Inputevents
    "starttime": "object",
    "endtime": "object",
    "amount": "float",
    "amountuom": "object",
    "rate": "float",
    "rateuom": "object",
    "orderid": "int",
    "linkorderid": "int",
    "ordercategoryname": "object",
    "secondaryordercategoryname": "object",
    "ordercomponenttypedescription": "object",
    "ordercategorydescription": "object",
    "patientweight": "float",
    "totalamount": "float",
    "totalamountuom": "object",
    "isopenbag": "int",
    "continueinnextdept": "int",
    "cancelreason": "int",
    "statusdescription": "object",
    "originalamount": "float",
    "originalrate": "float",
}


def handle_stay_group(stay_group):
    if stay_group.empty:  # Sometimes dask generates empty dataframes
        return
    stay_id = stay_group.name
    stay_group.to_csv(f"cache/simpledp/{stay_id}/chartevents.csv", index=False)


if __name__ == "__main__":
    chartevents = dd.read_csv(
        f"mimiciv/icu/chartevents.csv",
        assume_missing=True,
        blocksize=1e7,
        dtype=all_inclusive_dtypes,
    )

    # Only on first run
    # stay_ids = chartevents["stay_id"].unique().compute(scheduler="processes").to_list()

    # for sid in stay_ids:
    #     if not os.path.exists(f"cache/simpledp/{sid}"):
    #         os.makedirs(f"cache/simpledp/{sid}")

    # Do the agg
    chartevents.groupby("stay_id").apply(
        handle_stay_group, meta=pd.DataFrame()
    ).compute(scheduler="processes")
