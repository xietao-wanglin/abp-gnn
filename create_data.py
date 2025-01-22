from simulation import Simulation

import numpy as np
from tqdm import tqdm

np.random.seed(0)

if __name__ == '__main__':

    train_sims = 1000
    test_sims = 200
    for i in tqdm(range(train_sims), desc='Training Set'):
        N = np.random.randint(1, 100)
        rot_rate = 1
        sim = Simulation(N=N, v0=1, L_box=1.0, t_max=1, couple_radius=0.05, rot_rate=rot_rate, seed=i, timesteps=100)
        sim.solve_dynamics(method='RK45')
        _, loc = sim.get_solution()
        np.save(f'./data/simulation_train_{i}.npy', loc)
    
    for i in tqdm(range(test_sims), desc='Test Set'):
        N = np.random.randint(100, 120)
        rot_rate = 1
        sim = Simulation(N=N, v0=1, L_box=1.0, t_max=1, couple_radius=0.05, rot_rate=rot_rate, seed=i, timesteps=100)
        sim.solve_dynamics(method='RK45')
        _, loc = sim.get_solution()
        np.save(f'./data/simulation_test_{i}.npy', loc)
