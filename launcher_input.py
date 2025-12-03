from src.simulation import SparseWCA

import numpy as np
import argparse

def generate_state(n, box_length=1.0):
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return initial_state


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("power", help="Power k in n = 2^k")
    args = parser.parse_args()

    power = int(args.power)
    n = 2**power
    sigma = 0.04
    rho = 0.28
    box_length = sigma*np.sqrt(n*np.pi/rho)/2
    initial_state = generate_state(n=n)
    sim = SparseWCA(
        initial_state=initial_state,
        diffusion_r=0.0,
        diffusion_t=0.0,
        rot_rate=0.0,
        sigma=sigma,
        epsilon=0.1,
        timesteps=151,
        couple_radius=0.0,
        rot_couple=0.0,
        delta_t=1,
        box_length=box_length,
        record_every=1,
    )
    sim.solve_dynamics(method="RK45", debug=False)
    _times, loc = sim.get_solution()
    np.save(f"abp_analysis/abp_{n}.npy", loc[1:])
