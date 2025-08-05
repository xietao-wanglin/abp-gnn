from src.simulation import LennardJonesSimulation

import numpy as np


def generate_state(n, n_passive, n_boundary, box_length=1.0):
    rot_rate = np.ones(n)
    rot_rate[:n_passive] = np.zeros(n_passive)
    v0 = np.ones(n) * 0.1
    v0[:n_passive] = np.zeros(n_passive)
    rot_couple = np.zeros(n)
    particle_type = np.ones(n, dtype=int)
    particle_type[:n_boundary] = 0

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length / 4 + 0.25
    initial_state[1::3] = initial_state[1::3] * box_length / 4 + 0.25

    initial_state = initial_state.reshape(n, 3).T

    return rot_rate, v0, rot_couple, particle_type, initial_state


def chevron(n, n_boundary, box_length=1.0):
    x = np.linspace(0, 0.8, n_boundary) + 0.1
    y = 0.8 - np.abs(x - 0.5)
    z = np.zeros_like(x)
    boundary_state = np.vstack([y, x, z])
    rot_rate = np.ones(n + n_boundary) * 1
    v0 = np.ones(n + n_boundary) * 0.1
    rot_couple = np.zeros(n + n_boundary)
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T
    initial_state = np.concat([initial_state, boundary_state], axis=1)
    particle_type = np.hstack([np.ones(n), np.zeros(n_boundary)])
    return rot_rate, v0, rot_couple, particle_type, initial_state


def box(n, n_boundary, box_length=1.0, offset=0):
    t_values = np.linspace(0, 1, n_boundary, endpoint=False)

    boundary_points = []

    for t in t_values:
        perimeter_pos = t * 4 * box_length

        if perimeter_pos <= box_length:
            x = perimeter_pos + offset
            y = 0 + offset
        elif perimeter_pos <= 2 * box_length:
            x = box_length + offset
            y = perimeter_pos - box_length + offset
        elif perimeter_pos <= 3 * box_length:
            x = box_length - (perimeter_pos - 2 * box_length) + offset
            y = box_length + offset
        else:
            x = 0 + offset
            y = box_length - (perimeter_pos - 3 * box_length) + offset

        boundary_points.append([x, y, 0])

    boundary_points = np.array(boundary_points).T
    boundary_state = boundary_points

    rot_rate = np.ones(n + n_boundary) * 1
    v0 = np.ones(n + n_boundary) * 0.1
    rot_couple = np.zeros(n + n_boundary)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length + offset
    initial_state[1::3] = initial_state[1::3] * box_length + offset
    initial_state = initial_state.reshape(n, 3).T

    initial_state = np.concatenate([initial_state, boundary_state], axis=1)

    particle_type = np.hstack([np.ones(n), np.zeros(n_boundary)])

    return rot_rate, v0, rot_couple, particle_type, initial_state


def spiral(n, n_boundary, turns=2, box_length=1.0, center=0.5, width=0.35):
    theta = np.linspace(0, turns * 2 * np.pi, n_boundary)
    r = 0.05 + width * theta / (turns * 2 * np.pi)
    x = center + r * np.cos(theta)
    y = center + r * np.sin(theta)
    z = np.zeros_like(x)
    boundary_state = np.vstack([x, y, z])

    rot_rate = np.ones(n + n_boundary) * 1
    v0 = np.ones(n + n_boundary) * 0.1
    rot_couple = np.zeros(n + n_boundary)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length * 0.2 + box_length / 2 - 0.05
    initial_state[1::3] = initial_state[1::3] * box_length * 0.2 + box_length / 2 - 0.05
    initial_state = initial_state.reshape(n, 3).T
    initial_state = np.concatenate([initial_state, boundary_state], axis=1)

    particle_type = np.hstack([np.ones(n), np.zeros(n_boundary)])
    return rot_rate, v0, rot_couple, particle_type, initial_state


def circle(n, n_boundary, radius=0.4, box_length=1.0):
    theta = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    x = 0.5 + radius * np.cos(theta)
    y = 0.5 + radius * np.sin(theta)
    z = np.zeros_like(x)
    boundary_state = np.vstack([x, y, z])

    rot_rate = np.ones(n + n_boundary) * 1
    v0 = np.ones(n + n_boundary) * 0.1
    rot_couple = np.zeros(n + n_boundary)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length
    initial_state = initial_state.reshape(n, 3).T
    initial_state = np.concatenate([initial_state, boundary_state], axis=1)

    particle_type = np.hstack([np.ones(n), np.zeros(n_boundary)])
    return rot_rate, v0, rot_couple, particle_type, initial_state


def maze_channel(n, n_boundary, box_length=1.0, channel_width=0.08):
    y_vals = np.linspace(0, box_length, n_boundary // 2)

    boundary_points = []
    for y in y_vals:
        x_center = 0.5 + 0.25 * np.sin(3 * np.pi * y)

        boundary_points.append([x_center - channel_width, y, 0])

    for y in reversed(y_vals):
        x_center = 0.5 + 0.25 * np.sin(3 * np.pi * y)
        boundary_points.append([x_center + channel_width, y, 0])

    boundary_state = np.array(boundary_points).T

    rot_rate = np.ones(n + n_boundary) * 1
    v0 = np.ones(n + n_boundary) * 0.1
    rot_couple = np.zeros(n + n_boundary)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length
    initial_state = initial_state.reshape(n, 3).T
    initial_state = np.concatenate([initial_state, boundary_state], axis=1)

    particle_type = np.hstack([np.ones(n), np.zeros(n_boundary)])
    return rot_rate, v0, rot_couple, particle_type, initial_state


def star(n, n_boundary, n_points=5, box_length=1.0):
    theta = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)

    r = np.zeros_like(theta)
    for i, t in enumerate(theta):
        angle_per_point = 2 * np.pi / n_points
        point_phase = (t % angle_per_point) / angle_per_point

        if point_phase < 0.5:
            r[i] = 0.15 + 0.25 * (2 * point_phase)
        else:
            r[i] = 0.40 - 0.25 * (2 * (point_phase - 0.5))

    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    z = np.zeros_like(x)
    boundary_state = np.vstack([x, y, z])

    rot_rate = np.ones(n + n_boundary) * 1
    v0 = np.ones(n + n_boundary) * 0.1
    rot_couple = np.zeros(n + n_boundary)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length
    initial_state = initial_state.reshape(n, 3).T
    initial_state = np.concatenate([initial_state, boundary_state], axis=1)

    particle_type = np.hstack([np.ones(n), np.zeros(n_boundary)])
    return rot_rate, v0, rot_couple, particle_type, initial_state


def vortex_generator(n, n_boundary, box_length=1.0):
    centers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
    radius = 0.08

    boundary_points = []
    points_per_circle = n_boundary // 3

    for center in centers:
        theta = np.linspace(0, 2 * np.pi, points_per_circle, endpoint=False)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        for i in range(len(x)):
            boundary_points.append([x[i], y[i], 0])

    boundary_state = np.array(boundary_points).T

    rot_rate = np.ones(n + n_boundary) * 1
    v0 = np.ones(n + n_boundary) * 0.1
    rot_couple = np.zeros(n + n_boundary)

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length
    initial_state = initial_state.reshape(n, 3).T
    initial_state = np.concatenate([initial_state, boundary_state], axis=1)

    particle_type = np.hstack([np.ones(n), np.zeros(n_boundary)])
    return rot_rate, v0, rot_couple, particle_type, initial_state


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    np.random.seed(893)
    box_length = 1
    rot_rate, v0, rot_couple, particle_type, initial_state = generate_state(
        n=60, n_passive=0, n_boundary=0, box_length=box_length
    )

    sim = LennardJonesSimulation(
        initial_state=initial_state,
        v0=v0,
        box_length=box_length,
        delta_t=0.1,
        rot_couple=rot_couple,
        particle_type=particle_type,
        sigma=0.025,
        epsilon=0.1,
        rot_rate=rot_rate,
        timesteps=200,
        periodic=True,
        solver_times=False,
    )
    sim.solve_dynamics(method="RK45", debug=True)
    sim.create_animation()
