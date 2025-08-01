from src.simulation import RepulsiveSimulation

import numpy as np

np.random.seed(0)


def generate_state(n, n_passive, box_length=1.0):
    rot_rate = np.ones(n)
    rot_rate[:n_passive] = np.zeros(n_passive)
    v0 = np.ones(n) * 0.1
    v0[:n_passive] = np.zeros(n_passive)
    rot_couple = np.zeros(n)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return rot_rate, v0, rot_couple, initial_state


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    box_length = 1.0
    rot_rate, v0, rot_couple, initial_state = generate_state(
        n=100, n_passive=30, box_length=box_length
    )
    sim = RepulsiveSimulation(
        initial_state=initial_state,
        v0=v0,
        box_length=box_length,
        delta_t=0.1,
        rot_couple=rot_couple,
        sigma=0.025,
        epsilon=0.1,
        rot_rate=rot_rate,
        timesteps=200,
        periodic=True,
        solver_times=False,
    )
    sim.solve_dynamics(method="RK45", debug=True)
    sim.create_animation()
