from simulation import Simulation

import numpy as np

if __name__ == '__main__':

    N = 1000
    rot_rate = 1
    sim = Simulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, couple_radius=0.05, rot_rate=rot_rate, timesteps=100)
    sim.solve_dynamics(method='RK45')
    sim.create_animation(filename=None)
