from src.simulation import Simulation, InfiniteSimulation

import numpy as np
from tqdm import tqdm

import time

if __name__ == '__main__':

    num_of_particles = np.array([2**x for x in range(1, 14)])
    n_replic = 5
    rot_rate = 1
    times = np.zeros((num_of_particles.shape[0], n_replic))
    filename = 'scaling.dat'
    with open(filename, 'w') as f:
        f.close()
    for i, n in enumerate(tqdm(num_of_particles, desc='Power')):
        for replic in tqdm(range(n_replic), desc='Replic', leave=False):
            seed = np.random.randint(0, 1000000000)
            start = time.time()
            sim = Simulation(N=n, v0=0.1, L_box=1.0, delta_t=0.1, couple_radius=0.1, rot_rate=rot_rate, seed=seed, timesteps=100)
            sim.solve_dynamics(method='Euler')
            end = time.time()
            times[i, replic] = end-start
            _, loc = sim.get_solution_abs()
            np.save(f'./data_scaling/power_{replic}_{n}.npy', loc)
        with open(filename, 'ab') as f:
            np.savetxt(f, np.atleast_2d(times[i]), delimiter=',', newline = '\n')
