from simulation import Simulation, InfiniteSimulation, StiffSimulation

import numpy as np
from tqdm import tqdm

np.random.seed(0)

if __name__ == '__main__':

    train_sims = 1000
    test_sims = 200
    for i in tqdm(range(train_sims), desc='Training Set'): # 495 - 1000
        N = np.random.randint(50, 110)
        rot_rate = 1
        sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0.1, sigma=0.025, rot_rate=rot_rate, timesteps=100, seed=i)
        sim.solve_dynamics(method='Radau')
        _, loc = sim.get_solution_abs()
        np.save(f'./data_stiff_wide2/simulation_train_{i}.npy', loc)
    
    for i in tqdm(range(test_sims), desc='Test Set'):
        N = np.random.randint(50, 110)
        rot_rate = 1
        sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0.1,  sigma=0.025, rot_rate=rot_rate, timesteps=100, seed=98743*i+4500)
        sim.solve_dynamics(method='Radau')
        _, loc = sim.get_solution_abs()
        np.save(f'./data_stiff_wide2/simulation_test_{i}.npy', loc)
