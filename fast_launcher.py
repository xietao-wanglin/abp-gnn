from src.fast_simulation import WCA
import jax
import jax.numpy as jnp
import numpy as np
import math

jax.config.update("jax_enable_x64", True)


def generate_state(n, rho=0.28, sigma=0.04, rep=0, random=None):
    data = np.load(f"initial_conditions/n_{n}_{rep}.npy")
    initial_state = jnp.asarray(data)
    box_length = sigma * math.sqrt(n * math.pi / rho) / 2

    particle_type = jnp.ones(n)
    if random is not None:
        initial_state = np.random.random(3 * n)
        initial_state[2::3] *= 2 * np.pi
        initial_state[0::3] = initial_state[0::3] * box_length
        initial_state[1::3] = initial_state[1::3] * box_length

        initial_state = initial_state.reshape(n, 3).T
        initial_state = jnp.asarray(initial_state)

    return particle_type, box_length, initial_state


if __name__ == "__main__":
    reps = 20
    n = 128
    sigma = 0.04

    for i in range(0, reps):
        particle_type, box_length, initial_state = generate_state(
            n=n, rho=0.28, sigma=sigma, rep=i, random=None
        )
        sim = WCA(
            initial_state=initial_state,
            v0=0.1,
            rot_rate=0.1,
            epsilon=0.1,
            sigma=sigma,
            couple_radius=0.1,
            couple_strength=0.1,
            particle_type=particle_type,
            box_length=box_length,
        )
        out = sim.solve_dynamics(
            t_end=501, dt=1e-12, save_dt=1, debug=True, use_controller=True
        )
        res = np.array(out.block_until_ready())
        np.save(f"abp_analysis/vicsek_{n}_{i}.npy", res)

        jax.clear_caches()
