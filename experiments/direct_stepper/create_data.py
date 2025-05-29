from src.simulation import StiffSimulation

import numpy as np
from tqdm import tqdm

def generate_state(N, delta):

    while True:
        initial_state = np.random.random(3 * N)
        initial_state[2::3] *= 2 * np.pi

        initial_state = initial_state.reshape(N, 3).T

        x = initial_state[0]
        y = initial_state[1]

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        if np.min(dist) >= delta:
            return initial_state

if __name__ == '__main__':

    train_sims = 1000
    train_init = 0
    test_sims = 200
    test_init = 0
    long_test_sims = 4

    for i in tqdm(range(train_init, train_sims+train_init), desc='Training Set'):
        np.random.seed(i)
        N = np.random.randint(15, 30)
        rot_rate = 1
        initial_state = generate_state(N=N, delta=0.1)
        sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0, sigma=0.025, rot_rate=rot_rate, timesteps=20, seed=i, 
                              initial_state=initial_state, periodic=True)
        sim.solve_dynamics(method='Radau')
        times, loc = sim.get_solution_abs()
        np.save(f'./experiments/direct_stepper/data/simulation_train_{i}.npy', loc)
    
    for i in tqdm(range(test_init, test_sims+test_init), desc='Test Set'):
        np.random.seed(98743*i+4500)
        N = np.random.randint(15, 30)
        rot_rate = 1
        initial_state = generate_state(N=N, delta=0.1)
        sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0, sigma=0.025, rot_rate=rot_rate, timesteps=20, seed=98743*i+4500,
                              initial_state=initial_state, periodic=True)
        sim.solve_dynamics(method='Radau')
        times, loc = sim.get_solution_abs()
        np.save(f'./experiments/direct_stepper/data/simulation_test_{i}.npy', loc)

    for i in tqdm(range(4), desc='Long Test Set'):
        np.random.seed(983*i+2000)
        N = np.random.randint(15, 30)
        rot_rate = 1
        initial_state = generate_state(N=N, delta=0.1)
        sim = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0, sigma=0.025, rot_rate=rot_rate, timesteps=200, seed=983*i+2000,
                              initial_state=initial_state, periodic=True)
        sim.solve_dynamics(method='Radau')
        times, loc = sim.get_solution_abs()
        np.save(f'./experiments/direct_stepper/data/test_{i}.npy', loc)
