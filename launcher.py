from simulation import Simulation

import numpy as np

if __name__ == '__main__':

    N = 1000
    rot_rate = np.random.random(N)
    sim = Simulation(N=N, L_box=1.0, t_max=20, couple_radius=0.1, rot_rate=rot_rate)
    sim.solve_dynamics()
    sim.create_animation()
