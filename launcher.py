from src.simulation import AVM

import numpy as np


def generate_state(n, n_passive, n_boundary, box_length=1.0):
    rot_rate = np.ones(n) * 1
    rot_rate[:n_passive] = np.zeros(n_passive)
    v0 = np.ones(n) * 0.05
    v0[:n_passive] = np.zeros(n_passive)
    rot_couple = np.ones(n) * 0
    particle_type = np.ones(n, dtype=int)
    particle_type[:n_boundary] = 0

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return rot_rate, v0, rot_couple, particle_type, initial_state


def create_bidirectional_corridor(N=80, Lx=10.0, Ly=2.0, seed=42):
    np.random.seed(seed)
    half = N // 2

    # Positions
    xA = np.random.uniform(0, Lx / 2, half)
    yA = np.random.uniform(0, Ly, half) + 7

    xB = np.random.uniform(Lx / 2, Lx, half)
    yB = np.random.uniform(0, Ly, half) + 3

    # Directions (angles)
    # Group A → rightward (theta = 0)
    thetaA = np.zeros(half)
    # Group B → leftward (theta = π)
    thetaB = np.ones(half) * np.pi

    # Combine into one initial state [3, N]
    x = np.concatenate([xA, xB])
    y = np.concatenate([yA, yB])
    theta = np.concatenate([thetaA, thetaB])

    return np.vstack([x, y, theta]), Lx, Ly


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    np.random.seed(1)
    box_length = 5
    rot_rate, v0, rot_couple, particle_type, initial_state = generate_state(
        n=40, n_passive=0, n_boundary=0, box_length=box_length
    )
    initial_state, Lx, Ly = create_bidirectional_corridor(N=80, Lx=10, Ly=4)

    sim = AVM(
        initial_state=initial_state,
        v0=1.55,
        tau=0.3,
        T=1.06,
        D=0.1,
        k=3,
        t_a=1.0,
        r=0.18,
        box_length=Lx,
        timesteps=400,
        delta_t=0.05,
    )
    sim.solve_dynamics(method="Euler", debug=True)
    sim.create_animation(every=1)
