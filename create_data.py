from simulation import Simulation

import numpy as np
from tqdm import tqdm

np.random.seed(0)

if __name__ == '__main__':

    train_sims = 1000
    test_sims = 200
    for i in tqdm(range(train_sims), desc='Training Set'):
        N = np.random.randint(70, 250)
        rot_rate = 1
        sim = Simulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, couple_radius=0.1, rot_rate=rot_rate, seed=i, timesteps=100)
        sim.solve_dynamics(method='RK45')
        _, loc = sim.get_derivatives()
        np.save(f'./data_euler/simulation_train_{i}.npy', loc)
    
    for i in tqdm(range(test_sims), desc='Test Set'):
        N = np.random.randint(70, 250)
        rot_rate = 1
        sim = Simulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, couple_radius=0.1, rot_rate=rot_rate, seed=6000*i+1, timesteps=100)
        sim.solve_dynamics(method='RK45')
        _, loc = sim.get_derivatives()
        np.save(f'./data_euler/simulation_test_{i}.npy', loc)
