from simulation import Simulation, InfiniteSimulation, StiffSimulation

import numpy as np

if __name__ == '__main__':

    N = 110
    N_passive = 40
    rot_rate = np.ones(N)
    rot_rate[:N_passive] = np.zeros(N_passive)
    v0 = np.ones(N)*0.1
    v0[:N_passive] = np.zeros(N_passive)
    rot_couple = np.ones(N)*0.1
    rot_couple[:N_passive] = np.zeros(N_passive)
    sim = StiffSimulation(N=N, v0=v0, L_box=1.0, delta_t=0.1, rot_couple=rot_couple, epsilon=0.1, sigma=0.025, rot_rate=rot_rate, timesteps=200, seed=0)
    sim.solve_dynamics(method='RK45')
    sim.create_animation(filename=None, timesteps=200)
    _, loc = sim.get_solution_abs()
    #np.save(f'./data_col_norot/test_200.npy', loc)
