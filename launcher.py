from simulation import Simulation, InfiniteSimulation

import numpy as np

if __name__ == '__main__':

    N = 8192
    rot_rate = 1
    sim = Simulation(N=N, v0=0.1, L_box=1.0, delta_t=0.05, rot_couple=0.1, rot_rate=rot_rate, timesteps=200, seed=98743)
    sim.solve_dynamics(method='Euler')
    sim.create_animation(filename=None, timesteps=200)
    _, loc = sim.get_solution_abs()
    np.save(f'./data_scaling/test_8192_half.npy', loc)
