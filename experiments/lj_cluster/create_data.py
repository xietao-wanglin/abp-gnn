from src.simulation import LennardJonesSimulation

import numpy as np
from tqdm import tqdm


def generate_state(n, seed):
    np.random.seed(seed * 541234 + 6)
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    d1, d2, d3 = np.random.random(size=3)
    if d3 > 0.1:
        initial_state[0::3] = initial_state[0::3] * 0.25 + d1
        initial_state[1::3] = initial_state[1::3] * 0.25 + d2
    initial_state = initial_state.reshape(n, 3).T

    return initial_state


if __name__ == "__main__":
    train_sims = 1000
    train_init = 0
    test_sims = 200
    test_init = 0
    long_test_sims = 2

    v0 = 0.1
    box_length = 1.0
    delta_t = 0.1
    rot_couple = 0.0
    sigma = 0.025
    rot_rate = 1

    for i in tqdm(range(train_init, train_sims + train_init), desc="Training Set"):
        np.random.seed(i)
        n = np.random.randint(80, 120)
        initial_state = generate_state(n=n, seed=i)
        sim = LennardJonesSimulation(
            v0=v0,
            box_length=box_length,
            delta_t=delta_t,
            rot_couple=rot_couple,
            sigma=sigma,
            rot_rate=rot_rate,
            timesteps=60,
            seed=i,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="RK45")
        times, loc = sim.get_solution_abs()
        np.save(f"./experiments/lj_cluster/data/simulation_train_{i}.npy", loc)

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        np.random.seed(98743 * i + 4500)
        n = np.random.randint(80, 120)
        initial_state = generate_state(n=n, seed=i)
        sim = LennardJonesSimulation(
            v0=v0,
            box_length=box_length,
            delta_t=delta_t,
            rot_couple=rot_couple,
            sigma=sigma,
            rot_rate=rot_rate,
            timesteps=60,
            seed=98743 * i + 4500,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="RK45")
        times, loc = sim.get_solution_abs()
        np.save(f"./experiments/lj_cluster/data/simulation_test_{i}.npy", loc)

    for i in tqdm(range(4), desc="Long Test Set"):
        np.random.seed(983 * i + 2000)
        n = np.random.randint(80, 120)
        initial_state = generate_state(n=n, seed=i)
        sim = LennardJonesSimulation(
            v0=v0,
            box_length=box_length,
            delta_t=delta_t,
            rot_couple=rot_couple,
            sigma=sigma,
            rot_rate=rot_rate,
            timesteps=200,
            seed=983 * i + 2000,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="RK45")
        times, loc = sim.get_solution_abs()
        np.save(f"./experiments/lj_cluster/data/simulation_long_test_{i}.npy", loc)
