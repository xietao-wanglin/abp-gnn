from simulation import Simulation, InfiniteSimulation, StiffSimulation

import numpy as np

if __name__ == '__main__':

    N = 1000
    rot_rate = 1
    sim = InfiniteSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0.8, rot_rate=rot_rate, timesteps=200, seed=98743)
    sim.solve_dynamics(method='Euler')
    sim.create_animation(filename=None, timesteps=200)
    _, loc = sim.get_solution_abs()
    np.save(f'./data_inf/test_1000.npy', loc)
