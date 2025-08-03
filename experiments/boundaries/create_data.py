from src.simulation import RepulsiveSimulation

import numpy as np
from tqdm import tqdm


def generate_state(n, n_passive, n_boundary, box_length=1.0):
    rot_rate = np.ones(n)
    rot_rate[:n_passive] = np.zeros(n_passive)
    v0 = np.ones(n) * 0.1
    v0[:n_passive] = np.zeros(n_passive)
    rot_couple = np.zeros(n)
    particle_type = np.ones(n, dtype=int)
    particle_type[:n_boundary] = 0

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return rot_rate, v0, rot_couple, particle_type, initial_state


if __name__ == "__main__":
    train_sims = 1000
    train_init = 0
    test_sims = 200
    test_init = 0
    long_test_sims = 4

    for i in tqdm(range(train_init, train_sims + train_init), desc="Training Set"):
        np.random.seed(i)
        n = np.random.randint(60, 120)
        n_boundary = np.random.randint(1, n/2)
        box_length = 1
        rot_rate, v0, rot_couple, particle_type, initial_state = generate_state(
            n=n, n_passive=0, n_boundary=n_boundary, box_length=box_length
        )
        sim = RepulsiveSimulation(
            initial_state=initial_state,
            v0=v0,
            box_length=box_length,
            delta_t=0.1,
            rot_couple=rot_couple,
            particle_type=particle_type,
            sigma=0.025,
            epsilon=0.1,
            rot_rate=rot_rate,
            timesteps=60,
            periodic=True,
            solver_times=False,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"./experiments/boundaries/data/simulation_train_{i}.npy", loc)
        np.save(f"./experiments/boundaries/data/particle_train_{i}.npy", particle_type)

    for i in tqdm(range(test_init, test_sims + test_init), desc="Test Set"):
        np.random.seed(98743 * i + 4500)
        n = np.random.randint(60, 120)
        n_boundary = np.random.randint(1, n/2)
        box_length = 1
        rot_rate, v0, rot_couple, particle_type, initial_state = generate_state(
            n=n, n_passive=0, n_boundary=n_boundary, box_length=box_length
        )
        sim = RepulsiveSimulation(
            initial_state=initial_state,
            v0=v0,
            box_length=box_length,
            delta_t=0.1,
            rot_couple=rot_couple,
            particle_type=particle_type,
            sigma=0.025,
            epsilon=0.1,
            rot_rate=rot_rate,
            timesteps=60,
            periodic=True,
            solver_times=False,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"./experiments/boundaries/data/simulation_test_{i}.npy", loc)
        np.save(f"./experiments/boundaries/data/particle_test_{i}.npy", particle_type)

    for i in tqdm(range(4), desc="Long Test Set"):
        np.random.seed(983 * i + 2000)
        n = np.random.randint(60, 120)
        n_boundary = np.random.randint(1, n/2)
        box_length = 1
        rot_rate, v0, rot_couple, particle_type, initial_state = generate_state(
            n=n, n_passive=0, n_boundary=n_boundary, box_length=box_length
        )
        sim = RepulsiveSimulation(
            initial_state=initial_state,
            v0=v0,
            box_length=box_length,
            delta_t=0.1,
            rot_couple=rot_couple,
            particle_type=particle_type,
            sigma=0.025,
            epsilon=0.1,
            rot_rate=rot_rate,
            timesteps=400,
            periodic=True,
            solver_times=False,
        )
        sim.solve_dynamics(method="RK45")
        _times, loc = sim.get_solution_abs()
        np.save(f"./experiments/boundaries/data/test_{i}.npy", loc)
        np.save(f"./experiments/boundaries/data/particle_test_test_{i}.npy", particle_type)
