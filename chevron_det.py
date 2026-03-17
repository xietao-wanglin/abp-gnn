from src.fast_simulation import TorqueWCA
import jax
import jax.numpy as jnp
import numpy as np
import argparse

jax.config.update("jax_enable_x64", False)


def generate_state_chevron(
    n, box_length, delta, n_boundary=30, chevron_angle=60, arm_length=0.4
):
    half_angle_rad = np.deg2rad(chevron_angle / 2)
    center = np.array([box_length / 2, box_length / 2])

    arm_points = n_boundary // 2
    s = np.linspace(0, arm_length * box_length, arm_points)
    dx = s * np.sin(half_angle_rad)
    dy = s * np.cos(half_angle_rad)

    x_bound = np.concatenate([center[0] - dx, center[0] + dx])
    y_bound = np.concatenate([center[1] - dy, center[1] - dy])
    boundary_coords = np.stack([x_bound, y_bound], axis=1)

    x_free, y_free = [], []

    while len(x_free) < n:
        batch_size = n - len(x_free)
        candidates = np.random.random((batch_size, 2)) * box_length

        for cand in candidates:
            distances = np.linalg.norm(boundary_coords - cand, axis=1)

            if np.all(distances >= delta):
                x_free.append(cand[0])
                y_free.append(cand[1])

    x_free = np.array(x_free)
    y_free = np.array(y_free)
    theta_free = np.random.random(n) * 2 * np.pi

    theta_bound = np.random.random(len(x_bound)) * 2 * np.pi

    x_all = np.concatenate([x_bound, x_free])
    y_all = np.concatenate([y_bound, y_free])
    theta_all = np.concatenate([theta_bound, theta_free])

    initial_state = jnp.asarray(np.vstack([x_all, y_all, theta_all]))

    particle_type = np.ones(n + len(x_bound), dtype=int)
    particle_type[: len(x_bound)] = 0
    particle_type = jnp.asarray(particle_type)

    return initial_state, particle_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="index")
    parser.add_argument("--rep", type=int, default=0, help="rep")
    args = parser.parse_args()

    start_phi = 0
    end_phi = 160
    n_phi = 41

    phis = [start_phi + i * (end_phi - start_phi) / (n_phi - 1) for i in range(n_phi)]
    index = int(args.index)
    phi = phis[index]

    n = 200
    save_dt = 1
    sigma = 0.04
    box_length = 1
    reps = 20
    n_boundary = 20
    data = np.load(f"chevron/init/init_{index}.npz")["predictions"]
    particle_type = np.ones(n + n_boundary, dtype=int)
    particle_type[:n_boundary] = 0
    for rep in range(reps):
        initial_state = data[rep]
        sim = TorqueWCA(
            initial_state=initial_state,
            v0=0.1,
            rot_rate=0.0,
            epsilon=0.1,
            sigma=sigma,
            couple_radius=0.0,
            couple_strength=0.0,
            gamma=1,
            particle_type=particle_type,
            box_length=box_length,
        )
        out = sim.solve_dynamics(
            t_end=1000, dt=1e-6, save_dt=save_dt, debug=True, use_controller=True
        )
        res = np.array(out.block_until_ready())

        np.savez(
            f"chevron/data/det_{index}_{rep}",
            predictions=res,
            box_length=box_length,
            angle=phi,
            save_dt=save_dt,
        )
