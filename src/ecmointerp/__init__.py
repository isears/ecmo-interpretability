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
    cores_available: int


with open("mimicts/readme.txt", "r") as f:
    mimicts_config = f.readlines()

    for line in mimicts_config:
        if "timestep" in line:
            timestep_seconds = int(line.split("=")[-1])


config = Config(
    timestep_seconds=timestep_seconds,
    prediction_timesteps=1,
    model_path="/gpfs/home/isears1/Repos/tabnet-sepsis/cache/models/singleTst_2022-07-29_14:10:48",
    cores_available=len(os.sched_getaffinity(0)),
)
