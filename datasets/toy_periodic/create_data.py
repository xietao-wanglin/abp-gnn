from src.simulation import Toy
from src.utils import apply_periodic_boundary

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import json

def generate_state(n, box_length=1.0):
    rot_rate = np.ones(n) * 0
    rot_couple = np.zeros(n)
    epsilon = 0.05

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return (
        rot_rate,
        rot_couple,
        epsilon,
        initial_state,
    )


def compute_stats(script_dir, box_length=1.0):
    train_glob = sorted(glob(f"{script_dir}/data/simulation_train_*"))
    all_list = np.array([])
    for data in train_glob:
        data_arr = np.load(data)[:2]
        bcs = apply_periodic_boundary(
            torch.tensor(data_arr[::2]), dims=[box_length, box_length, 2 * torch.pi]
        ).numpy()
        pos_diff = data_arr[1::2] - bcs
        df = np.sqrt(pos_diff[:, 0] ** 2 + pos_diff[:, 1] ** 2).reshape(-1)
        all_list = np.hstack([all_list, df])
    df_describe = pd.DataFrame(all_list)
    mean, std = df_describe.mean()[0], df_describe.std()[0]
    return mean, std


if __name__ == "__main__":
    train_sims = 4000
    train_init = 0
    test_sims = 800
    test_init = 0
    long_test_sims = 1
    data_folder = "data"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, data_folder)

    os.makedirs(data_dir, exist_ok=True)

    for i in tqdm(range(train_init, train_sims + train_init), desc="Training Set"):
        np.random.seed(i)
        n = np.random.randint(10, 20)
        (
            rot_rate,
            rot_couple,
            epsilon,
            initial_state,
        ) = generate_state(n=n)
        sim = Toy(
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=2,
            epsilon=epsilon,
            initial_state=initial_state,
            boundary_type=(1, 1, 1),
            delta_t=0.1,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_train_{i}.npy", loc)

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        np.random.seed(98743 * i + 4500)
        n = np.random.randint(10, 20)
        (
            rot_rate,
            rot_couple,
            epsilon,
            initial_state,
        ) = generate_state(n=n)
        sim = Toy(
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=22,
            epsilon=epsilon,
            initial_state=initial_state,
            boundary_type=(1, 1, 1),
            delta_t=0.1,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_test_{i}.npy", loc)

    for i in tqdm(range(long_test_sims), desc="Long Test Set"):
        np.random.seed(983 * i + 2000)
        n = 20
        (
            rot_rate,
            rot_couple,
            epsilon,
            initial_state,
        ) = generate_state(n=n)
        sim = Toy(
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            epsilon=epsilon,
            timesteps=400,
            initial_state=initial_state,
            boundary_type=(1, 1, 1),
            delta_t=0.1,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_long_test_{i}.npy", loc)

    vel_mean, vel_std = compute_stats(script_dir)
    stats = {
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "angular_mean": 0,
        "angular_std": 0,
    }
    metadata_path = os.path.join(script_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(stats, f, indent=4)
