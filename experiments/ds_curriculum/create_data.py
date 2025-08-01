from src.simulation import RepulsiveSimulation

import numpy as np
from tqdm import tqdm


def generate_state(n, seed):
    np.random.seed(seed * 541234 + 6)
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    d1, d2, d3 = np.random.random(size=3)
    if d3 > 0.5:
        initial_state[0::3] = initial_state[0::3] * 0.25 + d1
        initial_state[1::3] = initial_state[1::3] * 0.25 + d2
    initial_state = initial_state.reshape(n, 3).T

    return initial_state


if __name__ == "__main__":
    train_sims = 1000
    train_init = 0
    test_sims = 200
    test_init = 0
    long_test_sims = 4

    for i in tqdm(range(train_init, train_sims + train_init), desc="Training Set"):
        np.random.seed(i)
        n = np.random.randint(15, 30)
        rot_rate = 1
        initial_state = generate_state(n=n, seed=i)
        sim = RepulsiveSimulation(
            v0=0.1,
            box_length=1.0,
            delta_t=0.1,
            rot_couple=0,
            sigma=0.025,
            rot_rate=rot_rate,
            timesteps=20,
            seed=i,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="Radau")
        times, loc = sim.get_solution_abs()
        np.save(f"./experiments/ds_curriculum/data/simulation_train_{i}.npy", loc)

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        np.random.seed(98743 * i + 4500)
        n = np.random.randint(15, 30)
        rot_rate = 1
        initial_state = generate_state(n=n, seed=i)
        sim = RepulsiveSimulation(
            v0=0.1,
            box_length=1.0,
            delta_t=0.1,
            rot_couple=0,
            sigma=0.025,
            rot_rate=rot_rate,
            timesteps=20,
            seed=98743 * i + 4500,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="Radau")
        times, loc = sim.get_solution_abs()
        np.save(f"./experiments/ds_curriculum/data/simulation_test_{i}.npy", loc)

    for i in tqdm(range(4), desc="Long Test Set"):
        np.random.seed(983 * i + 2000)
        n = np.random.randint(15, 30)
        rot_rate = 1
        initial_state = generate_state(n=n, seed=i)
        sim = RepulsiveSimulation(
            n=n,
            v0=0.1,
            box_length=1.0,
            delta_t=0.1,
            rot_couple=0,
            sigma=0.025,
            rot_rate=rot_rate,
            timesteps=200,
            seed=983 * i + 2000,
            initial_state=initial_state,
            periodic=True,
        )
        sim.solve_dynamics(method="Radau")
        times, loc = sim.get_solution_abs()
        np.save(f"./experiments/ds_curriculum/data/test_{i}.npy", loc)
