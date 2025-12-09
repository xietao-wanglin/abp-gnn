from src.fast_simulation import Toy
import jax
import jax.numpy as jnp
import numpy as np


def generate_state(key, n, box_length=1.0):
    key, subkey = jax.random.split(key)
    initial_state = jax.random.uniform(subkey, shape=(3 * n,))

    initial_state = initial_state.at[2::3].multiply(2 * jnp.pi)
    initial_state = initial_state.at[0::3].multiply(box_length)
    initial_state = initial_state.at[1::3].multiply(box_length)

    initial_state = initial_state.reshape(n, 3).T

    return initial_state, key


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    initial_state, _key = generate_state(key, n=30)
    sim = Toy(
        initial_state=initial_state,
        v0=0.1,
        rot_rate=0.0,
        epsilon=0.025,
    )
    out = sim.solve_dynamics(t_end=20, dt=1e-4, save_dt=0.1)
    np.save("test/toy.npy", out)
