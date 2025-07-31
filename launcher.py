from src.simulation import RepulsiveSimulation

import numpy as np
np.random.seed(0)

def generate_state(N, N_passive, L_box=1.0):
    rot_rate = np.ones(N)
    rot_rate[:N_passive] = np.zeros(N_passive)
    v0 = np.ones(N) * 0.1
    v0[:N_passive] = np.zeros(N_passive)
    rot_couple = np.zeros(N)

    initial_state = np.random.random(3 * N)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * L_box
    initial_state[1::3] = initial_state[1::3] * L_box

    initial_state = initial_state.reshape(N, 3).T

    return rot_rate, v0, rot_couple, initial_state

def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]

if __name__ == "__main__":
    rot_rate, v0, rot_couple, initial_state = generate_state(N=100, N_passive=30, L_box=1.0)
    sim = RepulsiveSimulation(
        N=100,
        v0=v0,
        L_box=1.0,
        delta_t=0.1,
        rot_couple=rot_couple,
        sigma=0.025,
        epsilon=0.1,
        rot_rate=rot_rate,
        timesteps=200,
        initial_state=initial_state,
        periodic=True,
        solver_times=False,
    )
    sim.solve_dynamics(method="RK45", debug=True)
    sim.create_animation()
