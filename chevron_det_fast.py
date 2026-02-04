from src.fast_simulation import WCA
import jax
import jax.numpy as jnp
import numpy as np
import math

jax.config.update("jax_enable_x64", True)

def generate_state_chevron(n, n_boundary=30, rho=0.14, sigma=0.04, chevron_angle=120 , arm_length=0.4):
    n_free = n

    chevron_angle = chevron_angle * math.pi / 180
    box_length = 1.3*sigma * math.sqrt(n * math.pi / rho) / 2
    x_free = np.random.random(n_free) * box_length
    y_free = np.random.random(n_free) * box_length
    theta_free = np.random.random(n_free) * 2 * np.pi

    center = np.array([box_length / 2, box_length / 2])
    arm_points = n_boundary // 2

    s = np.linspace(0, arm_length * box_length, arm_points)
    x_left = center[0] - s * np.cos(chevron_angle)
    y_left = center[1] - s * np.sin(chevron_angle)

    x_right = center[0] + s * np.cos(chevron_angle)
    y_right = center[1] - s * np.sin(chevron_angle)

    x_bound = np.concatenate([x_left, x_right])
    y_bound = np.concatenate([y_left, y_right])
    theta_bound = np.random.random(len(x_bound)) * 2 * np.pi

    x_all = np.concatenate([x_bound, x_free])
    y_all = np.concatenate([y_bound, y_free])
    theta_all = np.concatenate([theta_bound, theta_free])

    initial_state = np.vstack([x_all, y_all, theta_all])
    particle_type = np.ones(n+n_boundary, dtype=int)
    particle_type[: len(x_bound)] = 0

    initial_state = jnp.asarray(initial_state)
    particle_type = jnp.asarray(particle_type)

    return initial_state, particle_type, box_length


if __name__ == "__main__":
    n = 100
    save_dt = 1
    sigma = 0.02
    initial_state, particle_type, box_length = generate_state_chevron(
        n=n, rho=0.14, sigma=sigma, chevron_angle=45,
    )
    sim = WCA(
        initial_state=initial_state,
        v0=0.1,
        rot_rate=0.01,
        epsilon=0.01,
        sigma=sigma,
        couple_radius=0.0,
        couple_strength=0.0,
        particle_type=particle_type,
        box_length=box_length,
    )
    out = sim.solve_dynamics(
        t_end=101, dt=1e-12, save_dt=save_dt, debug=True, use_controller=True
    )
    res = np.array(out.block_until_ready())
    np.savez(
        "sim_out",
        predictions=res,
        box_length=box_length,
        initial_state=np.array(initial_state),
        dt=save_dt,
    )
