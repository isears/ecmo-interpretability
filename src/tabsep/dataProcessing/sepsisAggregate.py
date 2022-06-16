from mimic2ts import BaseAggregator, all_inclusive_dtypes
import dask.dataframe as dd
from typing import List
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
from tabsep import config


class SepsisAggregator(BaseAggregator):
    def __init__(
        self,
        mimic_path: str,
        dst_path: str,
        stay_ids: List[int],
        timestep_seconds: int = 3600,
        blocksize=25e6,
    ):

        self.data = dd.read_csv(
            f"{mimic_path}/derived/sepsis3.csv",
            assume_missing=True,
            blocksize=blocksize,
            dtype=all_inclusive_dtypes,
            parse_dates=["sofa_time", "suspected_infection_time"],
        )

        self.sepsis_feature_id = 77700

        super().__init__(
            mimic_path,
            dst_path,
            stay_ids,
            self.sepsis_feature_id,
            timestep_seconds=timestep_seconds,
            name="sepsis3",
        )

        ProgressBar(dt=60).register()

    def _parse_dates(self):
        def get_sepsis_onset(row):
            if row["suspected_infection_time"] > row["sofa_time"]:
                return int(row["suspected_infection_time"].timestamp())
            else:
                return int(row["sofa_time"].timestamp())

        self.data["event_epoch_time"] = self.data.apply(
            get_sepsis_onset, axis=1, meta=pd.Series([0])
        )

    def _value_parser(self, row):
        # If entry exists, there's sepsis
        return 1.0

    def _feature_id_parser(self, row):
        # Made up feature id
        # Convention to start derived feature ids with 777
        return self.sepsis_feature_id

    def _feature_combiner(self, tidx_group: pd.DataFrame):
        return tidx_group.max()


if __name__ == "__main__":
    stay_ids = pd.read_csv("cache/included_stayids.csv")["stay_id"].to_list()
    se = SepsisAggregator(
        "./mimiciv", "./mimicts", stay_ids, timestep_seconds=config.timestep_seconds,
    )

    se.do_agg()
