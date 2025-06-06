from src.simulation import StiffSimulation

import numpy as np
import time


def generate_state(N, delta):
    while True:
        initial_state = np.random.random(3 * N)
        initial_state[2::3] *= 2 * np.pi

        initial_state = initial_state.reshape(N, 3).T

        x = initial_state[0]
        y = initial_state[1]

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        if np.min(dist) >= delta:
            return initial_state


if __name__ == "__main__":
    N = 40
    N_passive = 0
    rot_rate = np.ones(N)
    rot_rate[:N_passive] = np.zeros(N_passive)
    v0 = np.ones(N) * 0.1
    v0[:N_passive] = np.zeros(N_passive)
    rot_couple = np.ones(N) * 0
    rot_couple[:N_passive] = np.zeros(N_passive)
    np.random.seed(0)
    initial_state = np.random.random(3 * N)
    initial_state[2::3] = initial_state[2::3] * 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * 1
    initial_state[1::3] = initial_state[1::3] * 1
    initial_state[2 : N_passive * 3 : 3] = initial_state[2 : N_passive * 3 : 3] * 0
    initial_state = initial_state.reshape(N, 3).T
    # sim_loc = './data/old/simulation_test_20.npy'
    # positions_true = np.load(sim_loc)
    # initial_state = positions_true[0]
    # initial_state = generate_state(N=N, delta=0.08)
    sim = StiffSimulation(
        N=N,
        v0=v0,
        L_box=1.0,
        delta_t=0.1,
        rot_couple=rot_couple,
        rot_rate=rot_rate,
        sigma=0.025,
        epsilon=0.1,
        couple_radius=0,
        timesteps=200,
        seed=0,
        periodic=False,
        solver_times=True,
        initial_state=initial_state,
    )
    start = time.time()
    sim.solve_dynamics(method="RK45", max_time=20)
    end = time.time()
    print(end - start)
    times, loc = sim.get_solution_abs()
    sim.create_animation(filename=None, timesteps=200, axis_offset=0)
    print(np.diff(times).mean(), np.median(np.diff(times)), len(times))
    # print(np.diff(times[6000:]).mean(), np.median(np.diff(times[6000:])), len(times))
    # np.save(f'./data/test_2.npy', loc)
