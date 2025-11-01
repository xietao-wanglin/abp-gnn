from src.simulation import WCA2

import numpy as np


def generate_state(n, box_length=1.0):
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return initial_state

def generate_state_poly(n, box_length=1.0):
    n_boundary = np.random.randint(1, 2)
    rot_rate = np.ones(n) * 1
    rot_couple = np.zeros(n)
    sigma = np.random.uniform(0.01, 0.14, size=(n,))

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T
    particle_type = np.ones(n, dtype=int)
    particle_type[:n_boundary] = 0

    return (
        rot_rate,
        rot_couple,
        sigma,
        initial_state,
        particle_type,
    )

def generate_state_chevron(n, box_length=1.0, chevron_angle=np.pi/4, arm_length=0.4):
    n_boundary = n // 5
    n_free = n - n_boundary

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
    particle_type = np.ones(n, dtype=int)
    particle_type[:len(x_bound)] = 0

    return initial_state, particle_type


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    np.random.seed(0)
    initial_state = generate_state(2000)
    sim = WCA2(
        v0=0.1,
        initial_state=initial_state,
        diffusion_r=0.0,
        rot_rate=0.1,
        sigma=0.04,
        epsilon=0.1,
        timesteps=410,
        couple_radius=0.1,
        rot_couple=0.1,
        delta_t=0.1,
        box_length=3,
    )
    sim.solve_dynamics(method="RK45", debug=True)
    _times, loc = sim.get_solution()
    np.save("vicsek.npy", loc[10:])
