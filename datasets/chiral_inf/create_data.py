from src.simulation import WCA

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import json


def generate_state(n, box_length=1, delta=0.0):
    rot_rate = np.ones(n) * 1
    rot_couple = np.zeros(n)
    sigma = 0.04
    epsilon = 0.1

    while True:
        initial_state = np.random.random(3 * n)
        initial_state[2::3] *= 2 * np.pi
        initial_state[0::3] *= box_length
        initial_state[1::3] *= box_length
        initial_state = initial_state.reshape(n, 3).T

        positions = initial_state[:2].T
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)

        if np.all(dists > delta):
            break

    return (
        rot_rate,
        rot_couple,
        sigma,
        epsilon,
        box_length,
        initial_state,
    )


def compute_stats(script_dir):
    sim_glob = sorted(glob(f"{script_dir}/data/simulation_train_*"))
    all_list = []
    for idx, data in enumerate(sim_glob):
        data_arr = np.load(data)[:2]
        pos_diff = data_arr[1::2] - data_arr[::2]
        df = np.sqrt(pos_diff[:, 0] ** 2 + pos_diff[:, 1] ** 2)

        all_list.append(df.reshape(-1))
    all_list = np.hstack(all_list)
    df_describe = pd.DataFrame(all_list)
    mean, std = df_describe.mean()[0], df_describe.std()[0]
    return mean, std


if __name__ == "__main__":
    train_sims = 0
    train_init = 0
    test_sims = 0
    test_init = 0
    long_test_sims = 0
    data_folder = "data"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, data_folder)

    os.makedirs(data_dir, exist_ok=True)

    for i in tqdm(range(train_init, train_sims), desc="Training Set"):
        np.random.seed(i)
        n = np.random.randint(5, 10)
        (
            rot_rate,
            rot_couple,
            sigma,
            epsilon,
            box_length,
            initial_state,
        ) = generate_state(n=n)
        sim = WCA(
            v0=3 * sigma,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=2,
            sigma=sigma,
            epsilon=epsilon,
            initial_state=initial_state,
            box_length=box_length,
            boundary_type=(0, 0, 1),
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_train_{i}.npy", loc)

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        np.random.seed(98743 * i + 4500)
        n = np.random.randint(5, 10)
        (
            rot_rate,
            rot_couple,
            sigma,
            epsilon,
            box_length,
            initial_state,
        ) = generate_state(n=n)
        sim = WCA(
            v0=3 * sigma,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=22,
            sigma=sigma,
            epsilon=epsilon,
            initial_state=initial_state,
            box_length=box_length,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_test_{i}.npy", loc)

    for i in tqdm(range(long_test_sims), desc="Long Test Set"):
        np.random.seed(983 * i + 2000)
        n = np.random.randint(25, 30)
        (
            rot_rate,
            rot_couple,
            sigma,
            epsilon,
            box_length,
            initial_state,
        ) = generate_state(n=n)
        sim = WCA(
            v0=3 * sigma,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            sigma=sigma,
            timesteps=400,
            initial_state=initial_state,
            box_length=box_length,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_long_test_{i}.npy", loc)

    vel_mean, vel_std = compute_stats(script_dir)
    stats = {
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "angular_mean": 0.1,
        "angular_std": 0,
    }
    metadata_path = os.path.join(script_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(stats, f, indent=4)
