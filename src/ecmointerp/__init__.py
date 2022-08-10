from dataclasses import dataclass
import os


@dataclass
class Config:
    # Timesteps in the mimicts database
    timestep_seconds: int
    # Number of timesteps before sepsis onset
    prediction_timesteps: int
    # Path to TST model to use in results generation
    model_path: str
    transfer_model_path: str
    cores_available: int


with open("mimicts/readme.txt", "r") as f:
    mimicts_config = f.readlines()

    for line in mimicts_config:
        if "timestep" in line:
            timestep_seconds = int(line.split("=")[-1])


config = Config(
    timestep_seconds=timestep_seconds,
    prediction_timesteps=1,
    model_path="cache/models/singleTst_2022-08-09_20:08:32",
    transfer_model_path="cache/models/transferTst_2022-08-08_20:16:30",
    cores_available=len(os.sched_getaffinity(0)),
)
