from src.simulation import WCA2

import numpy as np


def generate_state(n, box_length=1.0):
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return initial_state


def generate_state_with_grid_boundary(
    lc, n_boundary=400, sigma=0.04, central_fraction=0.25
):
    n_side = int(np.sqrt(n_boundary))
    box_length = n_side * lc

    x = np.linspace(0, box_length - lc, n_side) + lc / 2
    y = np.linspace(0, box_length - lc, n_side) + lc / 2
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_boundary], Y.flatten()[:n_boundary]

    central_width = box_length * central_fraction
    offset = (box_length - central_width) / 2

    while True:
        active_x = np.random.rand() * central_width + offset
        active_y = np.random.rand() * central_width + offset
        d = np.sqrt((X - active_x) ** 2 + (Y - active_y) ** 2)
        if np.all(d > 1 * sigma):
            break
    active_theta = np.random.rand() * 2 * np.pi

    all_x = np.concatenate([X, [active_x]])
    all_y = np.concatenate([Y, [active_y]])
    all_theta = np.concatenate([np.zeros(n_boundary), [active_theta]])

    n_total = n_boundary + 1
    particle_type = np.zeros(n_total, dtype=int)
    particle_type[-1] = 1

    initial_state = np.vstack([all_x, all_y, all_theta])
    return particle_type, initial_state, box_length


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    np.random.seed(0)
    lc = 0.115
    particle_type, initial_state, box_length = generate_state_with_grid_boundary(lc)
    # initial_state = generate_state(40)
    sim = WCA2(
        v0=0.04 * 3,
        initial_state=initial_state,
        rot_rate=1,
        sigma=0.04,
        timesteps=160,
        delta_t=0.1,
        rot_couple=0,
        diffusion_r=0.0,
        diffusion_t=0.0,
        particle_type=particle_type,
        box_length=box_length,
    )
    sim.solve_dynamics(method="RK45", debug=True)
    _times, loc = sim.get_solution()
    np.save(f"./analysis/data/lc-{lc}.npy", loc)
