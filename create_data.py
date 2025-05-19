from simulation import Simulation, InfiniteSimulation, StiffSimulation

import numpy as np
from tqdm import tqdm

np.random.seed(0)

if __name__ == '__main__':

    train_sims = 1000
    train_init = 0 
    test_sims = 200
    test_init = 0
    for i in tqdm(range(train_init, train_sims+train_init), desc='Training Set'):
        N = np.random.randint(20, 40)
        rot_rate = 1
        initial_state = np.random.random(3*N)
        initial_state[2::3] = initial_state[2::3]*2*np.pi
        initial_state[0::3] = initial_state[0::3]
        initial_state[1::3] = initial_state[1::3]
        initial_state = initial_state.reshape(N, 3).T
        sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0, sigma=0.025, rot_rate=rot_rate, timesteps=20, seed=i, 
                              initial_state=initial_state, periodic=False, solver_times=True)
        sim.solve_dynamics(method='Radau', max_time=1)
        times, loc = sim.get_solution_abs()
        np.save(f'./data/simulation_train_{i}.npy', loc)
        np.save(f'./data/times_train_{i}.npy', times)
    
    for i in tqdm(range(test_init, test_sims+test_init), desc='Test Set'):
        N = np.random.randint(20, 40)
        rot_rate = 1
        initial_state = np.random.random(3*N)
        initial_state[2::3] = initial_state[2::3]*2*np.pi
        initial_state[0::3] = initial_state[0::3]
        initial_state[1::3] = initial_state[1::3]
        initial_state = initial_state.reshape(N, 3).T
        sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0, sigma=0.025, rot_rate=rot_rate, timesteps=100, seed=98743*i+4500,
                              initial_state=initial_state, periodic=False, solver_times=True)
        sim.solve_dynamics(method='Radau', max_time=1)
        times, loc = sim.get_solution_abs()
        np.save(f'./data/simulation_test_{i}.npy', loc)
        np.save(f'./data/times_test_{i}.npy', times)
