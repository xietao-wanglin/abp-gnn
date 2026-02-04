from src.simulation import WCA

import numpy as np



def generate_state_chevron(n, n_boundary=50, box_length=1.0, chevron_angle=np.pi / 4, arm_length=0.4):
    n_free = n

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

    return initial_state, particle_type

if __name__ == "__main__":
    np.random.seed(0)
    n = 200
    sigma = 0.02
    rho = 0.28
    box_length = 2*sigma * np.sqrt(n * np.pi / rho) / 2
    initial_state, particle_type = generate_state_chevron(n=n, box_length=box_length)
    sim = WCA(
        initial_state=initial_state,
        v0=0.1,
        diffusion_r=0.0,
        diffusion_t=0.0,
        rot_rate=0.0,
        sigma=sigma,
        epsilon=0.01,
        timesteps=500,
        couple_radius=0.0,
        rot_couple=0.0,
        delta_t=0.1,
        box_length=box_length,
        record_every=1,
        start_record=0,
        particle_type=particle_type
    )
    sim.solve_dynamics(method="RK45", debug=True)
    _times, loc = sim.get_solution()
    sim.create_animation()
    #np.save(f"initial_conditions/n_{n}_{i}.npy", loc[-2])
