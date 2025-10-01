from src.simulation import LennardJones

import numpy as np


def generate_state(n, n_passive, n_boundary, box_length=1.0):
    rot_rate = np.ones(n) * 0
    rot_rate[:n_passive] = np.zeros(n_passive)
    v0 = np.ones(n) * 2
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


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    np.random.seed(893)
    box_length = 30
    rot_rate, v0, rot_couple, particle_type, initial_state = generate_state(
        n=40, n_passive=0, n_boundary=0, box_length=30
    )

    sim = LennardJones(
        initial_state=initial_state,
        v0=v0,
        box_length=30,
        delta_t=0.1,
        rot_couple=rot_couple,
        particle_type=particle_type,
        sigma=1,
        epsilon=1,
        rot_rate=rot_rate,
        timesteps=200,
        boundary_type=(1, 1, 1),
        diffusion_r=0.02,
        diffusion_t=0.01,
    )
    sim.solve_dynamics(method="RK45", debug=True)
    sim.create_animation(every=1)
