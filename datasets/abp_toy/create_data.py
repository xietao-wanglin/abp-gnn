from src.simulation import WCA
from src.utils import apply_periodic_boundary

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import json


def generate_state(box_length=1.0):
    delta_t = 0.001
    rot_rate = 0
    rot_couple = 0
    diffusion_r = 0.001
    diffusion_t = 0.001

    initial_state = np.array([[0.5 * box_length], [0.5 * box_length], [0.0]])

    return (
        delta_t,
        rot_rate,
        rot_couple,
        diffusion_r,
        diffusion_t,
        initial_state,
    )


def compute_stats(script_dir):
    train_glob = sorted(glob(f"{script_dir}/data/simulation_train_*"))
    all_list = np.array([])
    all_list_theta = np.array([])
    for data in train_glob:
        data_arr = np.load(data)
        bcs = apply_periodic_boundary(torch.tensor(data_arr[::2])).numpy()
        pos_diff = data_arr[1::2] - bcs
        df = np.sqrt(pos_diff[:, 0] ** 2 + pos_diff[:, 1] ** 2).reshape(-1)
        df_theta = np.sqrt(pos_diff[:, 2] ** 2).reshape(-1)
        all_list = np.hstack([all_list, df])
        all_list_theta = np.hstack([all_list, df_theta])
    df_describe = pd.DataFrame(all_list)
    df_describe_theta = pd.DataFrame(all_list_theta)
    mean, std = df_describe.mean()[0], df_describe.std()[0]
    angular_mean, angular_std = df_describe_theta.mean()[0], df_describe_theta.std()[0]
    return mean, std, angular_mean, angular_std


if __name__ == "__main__":
    train_sims = 50
    train_init = 0
    test_sims = 10
    test_init = 0
    long_test_sims = 1
    data_folder = "data"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, data_folder)

    os.makedirs(data_dir, exist_ok=True)

    for i in tqdm(range(train_init, train_sims + train_init), desc="Training Set"):
        (
            delta_t,
            rot_rate,
            rot_couple,
            diffusion_r,
            diffusion_t,
            initial_state,
        ) = generate_state()
        sim = WCA(
            delta_t=delta_t,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            diffusion_r=diffusion_r,
            diffusion_t=diffusion_t,
            timesteps=101,
            seed=i,
            initial_state=initial_state,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs(every=100)
        np.save(f"{script_dir}/{data_folder}/simulation_train_{i}.npy", loc)

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        (
            delta_t,
            rot_rate,
            rot_couple,
            diffusion_r,
            diffusion_t,
            initial_state,
        ) = generate_state()
        sim = WCA(
            delta_t=delta_t,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            diffusion_r=diffusion_r,
            diffusion_t=diffusion_t,
            timesteps=101,
            seed=98743 * i + 4500,
            initial_state=initial_state,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs(every=100)
        np.save(f"{script_dir}/{data_folder}/simulation_test_{i}.npy", loc)

    for i in tqdm(range(long_test_sims), desc="Long Test Set"):
        (
            delta_t,
            rot_rate,
            rot_couple,
            diffusion_r,
            diffusion_t,
            initial_state,
        ) = generate_state()
        sim = WCA(
            delta_t=delta_t,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            diffusion_r=diffusion_r,
            diffusion_t=diffusion_t,
            timesteps=401,
            seed=983 * i + 2000,
            initial_state=initial_state,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs(every=100)
        np.save(f"{script_dir}/{data_folder}/simulation_long_test_{i}.npy", loc)

    vel_mean, vel_std, angular_mean, angular_std = compute_stats(script_dir)
    stats = {
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "angular_mean": angular_mean,
        "angular_std": angular_std,
    }
    metadata_path = os.path.join(script_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(stats, f, indent=4)
