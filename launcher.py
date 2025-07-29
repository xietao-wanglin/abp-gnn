from src.simulation import RepulsiveSimulation

import numpy as np
import time


def generate_state(N, N_passive):
    rot_rate = np.ones(N)
    rot_rate[:N_passive] = np.zeros(N_passive)
    v0 = np.ones(N) * 0.1
    v0[:N_passive] = np.zeros(N_passive)
    rot_couple = np.ones(N) * 0.1
    rot_couple[:N_passive] = np.zeros(N_passive)

    initial_state = np.random.random(3 * N)
    initial_state[2::3] *= 2 * np.pi

    initial_state = initial_state.reshape(N, 3).T

    return rot_rate, v0, rot_couple, initial_state


if __name__ == "__main__":
    N = 2500
    N_passive = 0
    rot_rate = np.ones(N)
    rot_rate[:N_passive] = np.zeros(N_passive)
    v0 = np.ones(N) * 0.1
    v0[:N_passive] = np.zeros(N_passive)
    rot_couple = np.ones(N) * 0.1
    rot_couple[:N_passive] = np.zeros(N_passive)
    np.random.seed(0)
    initial_state = np.random.random(3 * N)
    initial_state[2::3] = initial_state[2::3] * 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * 5
    initial_state[1::3] = initial_state[1::3] * 5
    initial_state[2 : N_passive * 3 : 3] = initial_state[2 : N_passive * 3 : 3] * 0
    initial_state = initial_state.reshape(N, 3).T
    # sim_loc = './data/old/simulation_test_20.npy'
    # positions_true = np.load(sim_loc)
    # initial_state = positions_true[0]
    # initial_state = generate_state(N=N, delta=0.0)
    sim = RepulsiveSimulation(
        N=N,
        v0=v0,
        L_box=5.0,
        delta_t=0.1,
        rot_couple=rot_couple,
        sigma=0.025,
        epsilon=0.1,
        rot_rate=rot_rate,
        timesteps=200,
        initial_state=initial_state,
        periodic=True,
        solver_times=False,
    )
    start = time.time()
    sim.solve_dynamics(method="RK45", max_time=2)
    end = time.time()
    print(end - start)
    times, loc = sim.get_solution_abs()
    sim.create_animation(filename=None, timesteps=len(times), axis_offset=0)
    offset = 0
    print(
        np.diff(times[offset:]).mean(),
        np.median(np.diff(times[offset:])),
        np.min(np.diff(times[offset:])),
        np.max(np.diff(times[offset:])),
        len(times[offset:]),
    )
    print("Steps until 1e-3")
    print((np.diff(times) < 1e-3).sum())
    print("Time until 1e-3")
    print(np.cumsum(times)[(np.diff(times) < 1e-3).sum()])
    # np.save(f'./data/test_2.npy', loc)
