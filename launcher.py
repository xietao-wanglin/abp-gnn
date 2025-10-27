from src.simulation import WCA

import numpy as np


def generate_state(n, box_length=1.0):
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return initial_state


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    np.random.seed(0)
    initial_state = generate_state(100)
    sim = WCA(
        v0=0.1,
        initial_state=initial_state,
        rot_rate=0.1,
        sigma=0.04,
        epsilon=0.1,
        timesteps=300,
        diffusion_r=0.01,
        delta_t=0.1,
        box_length=1,
    )
    sim.solve_dynamics(method="RK45", debug=True)
    sim.create_animation()
