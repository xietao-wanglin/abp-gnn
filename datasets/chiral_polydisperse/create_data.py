from src.simulation import WCA, ParticleType
from src.utils import apply_periodic_boundary

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import json


def generate_state(n, box_length=1.0):
    n_boundary = np.random.randint(1, n)
    rot_rate = np.ones(n) * 1
    rot_couple = np.zeros(n)
    sigma = np.random.uniform(0.01, 0.07, size=(n,))

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T
    particle_type = np.ones(n, dtype=int)
    particle_type[:n_boundary] = 0

    return (
        rot_rate,
        rot_couple,
        sigma,
        initial_state,
        particle_type,
    )


def compute_stats(script_dir):
    sim_glob = sorted(glob(f"{script_dir}/data/simulation_train_*"))
    particle_glob = sorted(glob(f"{script_dir}/data/particle_train_*"))
    all_list = []
    for idx, data in enumerate(sim_glob):
        data_arr = np.load(data)
        particle_arr = np.load(particle_glob[idx])
        bcs = apply_periodic_boundary(torch.tensor(data_arr[::2])).numpy()
        pos_diff = data_arr[1::2] - bcs
        df = np.sqrt(pos_diff[:, 0] ** 2 + pos_diff[:, 1] ** 2)

        mask = particle_arr == ParticleType.ACTIVE
        df_filtered = df[:, mask]
        all_list.append(df_filtered.reshape(-1))
    all_list = np.hstack(all_list)
    df_describe = pd.DataFrame(all_list)
    mean, std = df_describe.mean()[0], df_describe.std()[0]
    return mean, std


if __name__ == "__main__":
    train_sims = 1000
    train_init = 0
    test_sims = 200
    test_init = 0
    long_test_sims = 4
    data_folder = "data"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, data_folder)

    os.makedirs(data_dir, exist_ok=True)

    for i in tqdm(range(train_init, train_sims + train_init), desc="Training Set"):
        np.random.seed(i)
        n = np.random.randint(15, 30)
        (
            rot_rate,
            rot_couple,
            sigma,
            initial_state,
            particle_type,
        ) = generate_state(n=n)
        sim = WCA(
            v0=3 * sigma,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=81,
            sigma=sigma,
            particle_type=particle_type,
            initial_state=initial_state,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_train_{i}.npy", loc)
        np.save(f"{script_dir}/{data_folder}/particle_train_{i}.npy", particle_type)
        np.save(
            f"{script_dir}/{data_folder}/features_train_{i}.npy", sigma[np.newaxis, ...]
        )

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        np.random.seed(98743 * i + 4500)
        n = np.random.randint(15, 30)
        (
            rot_rate,
            rot_couple,
            sigma,
            initial_state,
            particle_type,
        ) = generate_state(n=n)
        sim = WCA(
            v0=3 * sigma,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=81,
            sigma=sigma,
            particle_type=particle_type,
            initial_state=initial_state,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_test_{i}.npy", loc)
        np.save(f"{script_dir}/{data_folder}/particle_test_{i}.npy", particle_type)
        np.save(
            f"{script_dir}/{data_folder}/features_test_{i}.npy", sigma[np.newaxis, ...]
        )

    for i in tqdm(range(long_test_sims), desc="Long Test Set"):
        np.random.seed(983 * i + 2000)
        n = np.random.randint(25, 30)
        (
            rot_rate,
            rot_couple,
            sigma,
            initial_state,
            particle_type,
        ) = generate_state(n=n)
        sim = WCA(
            v0=3 * sigma,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            sigma=sigma,
            particle_type=particle_type,
            timesteps=400,
            initial_state=initial_state,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"{script_dir}/{data_folder}/simulation_long_test_{i}.npy", loc)
        np.save(f"{script_dir}/{data_folder}/particle_long_test_{i}.npy", particle_type)
        np.save(
            f"{script_dir}/{data_folder}/features_long_test_{i}.npy",
            sigma[np.newaxis, ...],
        )

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
