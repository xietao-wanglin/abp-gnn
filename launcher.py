from src.simulation import WCA

import numpy as np



def generate_state_with_grid_boundary(n_boundary, box_length=1.0, sigma=0.02):
    """
    Create a square lattice of fixed obstacles and one active chiral particle.
    """
    n_side = int(np.ceil(np.sqrt(n_boundary)))
    lc = box_length / n_side  # lattice spacing

    # grid of obstacle centers
    x = np.linspace(0, box_length - lc, n_side) + lc / 2
    y = np.linspace(0, box_length - lc, n_side) + lc / 2
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_boundary], Y.flatten()[:n_boundary]

    # choose random position for the active particle without overlap
    while True:
        active_x = np.random.rand() * box_length
        active_y = np.random.rand() * box_length
        d = np.sqrt((X - active_x)**2 + (Y - active_y)**2)
        if np.all(d > 1.5 * sigma):
            break
    active_theta = np.random.rand() * 2 * np.pi

    # combine all
    all_x = np.concatenate([X, [active_x]])
    all_y = np.concatenate([Y, [active_y]])
    all_theta = np.concatenate([np.zeros(n_boundary), [active_theta]])

    n_total = n_boundary + 1

    rot_rate = np.zeros(n_total)
    v0 = np.zeros(n_total)
    particle_type = np.zeros(n_total, dtype=int)

    # single chiral active particle
    rot_rate[-1] = 1.0
    v0[-1] = 3 * sigma  # matches paper: v = 3σ
    particle_type[-1] = 1  # active = 1, obstacles = 0

    initial_state = np.vstack([all_x, all_y, all_theta])
    return particle_type, initial_state


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
    box_length = 1
    rho = 0.095
    N = 100
    sigma = 2*box_length*np.sqrt(rho/(N*np.pi))
    v0 = 3*sigma
    particle_type, initial_state = generate_state_with_grid_boundary(
        n_boundary=N, box_length=box_length, sigma=sigma
    )

    print("sigma =", sigma)
    print("lattice spacing lc =", box_length / np.sqrt(N))
    print("orbit radius R0 =", 3*sigma)
    print("ratio lc/R0 =", (box_length/np.sqrt(N)) / (3*sigma))

    sim = WCA(
        initial_state=initial_state,
        sigma=0.04,
        v0=v0,
        box_length=box_length,
        timesteps=4000,
        delta_t=0.1,
    )
    sim.solve_dynamics(method="Euler", debug=True)
    sim.create_animation(every=10, color_type=False, trail_length=0)
