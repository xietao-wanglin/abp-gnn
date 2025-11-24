from src.simulation import WCA

import numpy as np

def generate_state_with_grid_boundary(
    lc, n_boundary=400, sigma=0.04
):
    n_side = int(np.sqrt(n_boundary))
    box_length = n_side * lc

    x = np.linspace(0, box_length - lc, n_side) + lc / 2
    y = np.linspace(0, box_length - lc, n_side) + lc / 2
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_boundary], Y.flatten()[:n_boundary]

    while True:
        active_x = np.random.rand() * box_length
        active_y = np.random.rand() * box_length
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


if __name__ == "__main__":
    lcs = [0.2, 0.1253, 0.115, 0.11098, 0.09, 0.15, 0.1]
    n_replications = 2
    for lc in lcs:
        for i in range(n_replications):
            particle_type, initial_state, box_length = generate_state_with_grid_boundary(lc=lc)
            density = round(100 * 100 * np.pi * (0.04) ** 2 / (box_length) ** 2, 2)
            sim = WCA(
                v0=3*0.04,
                initial_state=initial_state,
                diffusion_r=0.0,
                diffusion_t=0.0,
                rot_rate=1.0,
                sigma=0.04,
                epsilon=1.0,
                timesteps=16000,
                couple_radius=0.0,
                rot_couple=0.0,
                delta_t=0.1,
                box_length=box_length,
                particle_type=particle_type,
            )
            sim.solve_dynamics(method="RK45", debug=False)
            _times, loc = sim.get_solution()
            np.save(f"./lattice/density_{density}_i.npy")
