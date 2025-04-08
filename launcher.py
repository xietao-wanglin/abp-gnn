from simulation import Simulation, InfiniteSimulation, StiffSimulation

import numpy as np

if __name__ == '__main__':

    N = 200
    rot_rate = 1
    sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.5, rot_couple=0, sigma=0.025, epsilon=0.1, rot_rate=rot_rate, timesteps=200, seed=0)
    sim.solve_dynamics(method='RK45')
    sim.create_animation(filename=None, timesteps=200)
    _, loc = sim.get_solution_abs()
    np.save(f'./data_col_/test_200_fast.npy', loc)
