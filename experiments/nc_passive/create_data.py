from src.simulation import RepulsiveSimulation

import numpy as np
from tqdm import tqdm


def generate_state(n, n_passive):
    rot_rate = np.ones(n)
    rot_rate[:n_passive] = np.zeros(n_passive)
    v0 = np.ones(n) * 0.1
    v0[:n_passive] = np.zeros(n_passive)
    rot_couple = np.zeros(n)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi

    initial_state = initial_state.reshape(n, 3).T

    return rot_rate, v0, rot_couple, initial_state


if __name__ == "__main__":
    train_sims = 1000
    train_init = 0
    test_sims = 200
    test_init = 0
    long_test_sims = 4

    for i in tqdm(range(train_init, train_sims + train_init), desc="Training Set"):
        np.random.seed(i)
        n = np.random.randint(60, 120)
        n_passive = np.random.randint(0, n)
        rot_rate, v0, rot_couple, initial_state = generate_state(
            n=n, n_passive=n_passive
        )
        sim = RepulsiveSimulation(
            v0=v0,
            box_length=1.0,
            delta_t=0.1,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=100,
            epsilon=0.1,
            sigma=0.025,
            seed=i,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="RK45")
        times, loc = sim.get_extended_solutions()
        np.save(f"./experiments/nc_passive/data/simulation_train_{i}.npy", loc)

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        np.random.seed(98743 * i + 4500)
        n = np.random.randint(60, 120)
        n_passive = np.random.randint(0, n)
        rot_rate, v0, rot_couple, initial_state = generate_state(
            n=n, n_passive=n_passive
        )
        sim = RepulsiveSimulation(
            v0=v0,
            box_length=1.0,
            delta_t=0.1,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            timesteps=100,
            epsilon=0.1,
            sigma=0.025,
            seed=98743 * i + 4500,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="RK45")
        times, loc = sim.get_extended_solutions()
        np.save(f"./experiments/nc_passive/data/simulation_test_{i}.npy", loc)

    for i in tqdm(range(4), desc="Long Test Set"):
        np.random.seed(983 * i + 2000)
        n = np.random.randint(60, 120)
        n_passive = np.random.randint(0, n)
        rot_rate, v0, rot_couple, initial_state = generate_state(
            n=n, n_passive=n_passive
        )
        sim = RepulsiveSimulation(
            v0=v0,
            box_length=1.0,
            delta_t=0.1,
            rot_couple=rot_couple,
            rot_rate=rot_rate,
            epsilon=0.1,
            sigma=0.025,
            timesteps=400,
            seed=983 * i + 2000,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="RK45")
        times, loc = sim.get_extended_solutions()
        np.save(f"./experiments/nc_passive/data/test_{i}.npy", loc)
