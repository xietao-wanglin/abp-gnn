from src.fast_simulation import BoundaryWCA

import jax
import jax.numpy as jnp
import numpy as np
import argparse

import equinox as eqx

jax.config.update("jax_enable_x64", True)


@eqx.filter_jit
def run_sim(model, wrap, t_end, save_dt):
    return model.solve_dynamics(
        t_end=t_end,
        wrap=wrap,
        dt=1e-4,
        save_dt=save_dt,
        debug=True,
        use_controller=False,
    )


def generate_state_with_grid_boundary(phi, n_active=10, n_boundary=400, sigma=0.04):
    box_length = np.sqrt(100 * 100 * np.pi * (sigma) ** 2 / phi)
    n_side = int(np.sqrt(n_boundary))
    lc = box_length / 20

    x = np.linspace(0, box_length - lc, n_side) + lc / 2
    y = np.linspace(0, box_length - lc, n_side) + lc / 2
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_boundary], Y.flatten()[:n_boundary]

    active_x_list = []
    active_y_list = []
    active_theta_list = []

    while len(active_x_list) < n_active:
        candidate_x = np.random.rand() * box_length
        candidate_y = np.random.rand() * box_length

        d_to_boundary = np.sqrt((X - candidate_x) ** 2 + (Y - candidate_y) ** 2)

        if np.all(d_to_boundary > sigma):
            active_x_list.append(candidate_x)
            active_y_list.append(candidate_y)
            active_theta_list.append(np.random.rand() * 2 * np.pi)

    all_x = np.concatenate([X, active_x_list])
    all_y = np.concatenate([Y, active_y_list])
    all_theta = np.concatenate([np.zeros(n_boundary), active_theta_list])

    n_total = n_boundary + n_active
    particle_type = np.zeros(n_total, dtype=int)
    particle_type[n_boundary:] = 1

    initial_state = jnp.array(np.vstack([all_x, all_y, all_theta]))
    particle_type = jnp.array(particle_type)

    return particle_type, box_length, initial_state


if __name__ == "__main__":
    n = 200
    save_dt = 1
    t_end = 1600

    sigma = 1
    v0 = 3 * sigma
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="index")
    args = parser.parse_args()
    start_phi = 1
    end_phi = 21
    n_phi = 81

    phis = [start_phi + i * (end_phi - start_phi) / (n_phi - 1) for i in range(n_phi)]
    index = int(args.index)
    phi = phis[index]

    particle_type, box_length, initial_state = generate_state_with_grid_boundary(
        phi=phi, n_active=n, n_boundary=400
    )
    sim = BoundaryWCA(
        initial_state=initial_state,
        v0=v0,
        rot_rate=1.0,
        epsilon=0.1,
        sigma=sigma,
        couple_radius=0.0,
        couple_strength=0.0,
        particle_type=particle_type,
        box_length=box_length,
    )
    out = run_sim(sim, wrap=False, t_end=t_end, save_dt=save_dt)
    res = np.array(out)
    np.savez(
        "sim.out.npz",
        predictions=res,
        box_length=box_length,
        initial_state=np.array(initial_state),
        phi=phi,
        dt=save_dt,
    )
