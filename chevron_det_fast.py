from src.fast_simulation import *
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

def generate_state_chevron(n, box_length, n_boundary=30, chevron_angle=60, arm_length=0.4):
    n_free = n

    half_angle_rad = np.deg2rad(chevron_angle / 2)
    
    x_free = np.random.random(n_free) * box_length
    y_free = np.random.random(n_free) * box_length
    theta_free = np.random.random(n_free) * 2 * np.pi

    center = np.array([box_length / 2, box_length / 2])
    arm_points = n_boundary // 2
    s = np.linspace(0, arm_length * box_length, arm_points)
    dx = s * np.sin(half_angle_rad)
    dy = s * np.cos(half_angle_rad)
    x_left = center[0] - dx
    y_left = center[1] - dy 

    x_right = center[0] + dx
    y_right = center[1] - dy 

    x_bound = np.concatenate([x_left, x_right])
    y_bound = np.concatenate([y_left, y_right])
    theta_bound = np.random.random(len(x_bound)) * 2 * np.pi

    x_all = np.concatenate([x_bound, x_free])
    y_all = np.concatenate([y_bound, y_free])
    theta_all = np.concatenate([theta_bound, theta_free])

    initial_state = np.vstack([x_all, y_all, theta_all])
    particle_type = np.ones(n + n_boundary, dtype=int)
    particle_type[: len(x_bound)] = 0

    initial_state = jnp.asarray(initial_state)
    particle_type = jnp.asarray(particle_type)

    return initial_state, particle_type


if __name__ == "__main__":
    np.random.seed(1)
    n = 200
    save_dt = 1
    sigma = 0.02
    box_length = 0.6
    initial_state, particle_type= generate_state_chevron(
        n=n, chevron_angle=40, n_boundary=40, box_length=box_length
    )
    sim = TorqueWCA(
        initial_state=initial_state,
        v0=0.1,
        rot_rate=0.0,
        epsilon=0.001,
        sigma=sigma,
        couple_radius=0.0,
        couple_strength=0.0,
        gamma=1,
        particle_type=particle_type,
        box_length=box_length,
    )
    out = sim.solve_dynamics(
        t_end=300, dt=1e-12, save_dt=save_dt, debug=True, use_controller=True
    )
    res = np.array(out.block_until_ready())
    np.savez(
        "sim_out",
        predictions=res,
        box_length=box_length,
        initial_state=np.array(initial_state),
        dt=save_dt,
    )
