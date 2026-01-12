from src.fast_simulation import WCA

import jax
import jax.numpy as jnp
import numpy as np
import argparse

import csv
import equinox as eqx

jax.config.update("jax_enable_x64", False)

@eqx.filter_jit
def run_fast_sim(sim_obj, t_end, dt, save_dt):
    return sim_obj.solve_dynamics(
        t_end=t_end, 
        dt=dt, 
        save_dt=save_dt, 
        debug=True,
        use_controller=True
    )

def unwrap(traj, box_length):
    unwrapped = np.zeros_like(traj)
    unwrapped[0] = traj[0]

    for i in range(1, len(traj)):
        dx = traj[i] - traj[i-1]
        dx -= np.round(dx / box_length) * box_length
        unwrapped[i] = unwrapped[i-1] + dx

    return unwrapped

def generate_state_with_grid_boundary(lc, n_boundary=400, sigma=0.04):
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
    initial_state = jnp.asarray(initial_state)
    particle_type = jnp.asarray(particle_type)
    return particle_type, initial_state, box_length


if __name__ == "__main__":

    sigma = 0.04
    v0 = 3 * sigma
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="index")
    args = parser.parse_args()
    start_lc = 0.095
    end_lc = 0.2
    n_lc = 80

    lcs = [start_lc + i * (end_lc - start_lc) / (n_lc - 1) for i in range(n_lc)]
    index = int(args.index)
    lc = lcs[index]
    n_replications = 2
    all_disp = []
    for i in range(n_replications):
        particle_type, initial_state, box_length = (
            generate_state_with_grid_boundary(lc=lc)
        )
        density = round(100 * 100 * np.pi * (0.04) ** 2 / (box_length) ** 2, 2)
        sim = WCA(
            initial_state=initial_state,
            v0=v0,
            rot_rate=1.0,
            epsilon=1.0,
            sigma=sigma,
            couple_radius=0.0,
            couple_strength=0.0,
            particle_type=particle_type,
            box_length=box_length,
        )
        out = run_fast_sim(sim, 1600, 1e-12, 1)
        res = np.array(out)
        traj = res[:-1, :2, -1]
        traj_unwrap = unwrap(traj, box_length)
        disp = traj_unwrap - traj_unwrap[0]
        msd = np.sum(disp**2, axis=1)
        all_disp.append(msd)
    
    all_disp = np.vstack(all_disp)
    msd_mean = all_disp.mean(axis=0)
    dt = 1
    time = np.arange(len(msd_mean)) * dt
    start = 1200
    t_fit = time[start:]
    msd_fit = msd_mean[start:]
    slope, intercept = np.polyfit(t_fit, msd_fit, 1)
    D = slope / 4.0
    D_adj = D/(sigma*v0)

    save_path = "lattice/data.csv"
    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([density, D_adj])
