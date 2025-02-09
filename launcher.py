from simulation import Simulation

import numpy as np

if __name__ == '__main__':

    N = 180
    rot_rate = 1
    sim = Simulation(N=N, v0=0.1, L_box=1.0, delta_t=0.01, couple_radius=0.1, rot_rate=rot_rate, timesteps=10000, seed=9876543)
    sim.solve_dynamics(method='RK45')
    sim.create_animation(filename=None)
    _, loc = sim.get_solution()
    np.save(f'./data/vsmall_t.npy', loc)
