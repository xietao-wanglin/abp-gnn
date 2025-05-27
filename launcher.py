from simulation import Simulation, InfiniteSimulation, StiffSimulation
from create_data import generate_state

import numpy as np

if __name__ == '__main__':

    N = 20
    N_passive = 0
    rot_rate = np.ones(N)
    rot_rate[:N_passive] = np.zeros(N_passive)
    v0 = np.ones(N)*0.1
    v0[:N_passive] = np.zeros(N_passive)
    rot_couple = np.ones(N)*0
    rot_couple[:N_passive] = np.zeros(N_passive)
    initial_state = np.random.random(3*N)
    initial_state[2::3] = initial_state[2::3]*2*np.pi
    initial_state[0::3] = initial_state[0::3]
    initial_state[1::3] = initial_state[1::3]
    initial_state[2:N_passive*3:3] = initial_state[2:N_passive*3:3]*0
    initial_state = initial_state.reshape(N, 3).T
    initial_state = generate_state(N=N, delta=0.1)
    sim = StiffSimulation(N=N, v0=v0, L_box=1.0,
                           delta_t=0.1, 
                           rot_couple=rot_couple, 
                           rot_rate=rot_rate, 
                           sigma=0.025, couple_radius=0, timesteps=100, seed=0, periodic=False, solver_times=False,
                             initial_state=initial_state)
    sim.solve_dynamics(method='Radau', max_time=0.1)
    times, loc = sim.get_solution_abs()
    sim.create_animation(filename=None, timesteps=len(times), axis_offset=0)
    print(times, len(times))
    #np.save(f'./data/test_2.npy', loc)
