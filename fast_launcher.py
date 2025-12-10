from src.fast_simulation import WCA
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def generate_state(key, n, box_length=1.0):
    key, subkey = jax.random.split(key)
    initial_state = jax.random.uniform(subkey, shape=(3 * n,))

    initial_state = initial_state.at[2::3].multiply(2 * jnp.pi)
    initial_state = initial_state.at[0::3].multiply(box_length)
    initial_state = initial_state.at[1::3].multiply(box_length)

    initial_state = initial_state.reshape(n, 3).T

    particle_type = jnp.ones(n)

    return particle_type, initial_state, key


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    particle_type, initial_state, _key = generate_state(key, n=30)
    sim = WCA(
        initial_state=initial_state,
        v0=0.1,
        rot_rate=0.0,
        epsilon=0.1,
        sigma=0.04,
        couple_radius=0.1,
        couple_strength=0.1,
        particle_type=particle_type,
    )
    out = sim.solve_dynamics(t_end=40, dt=1e-4, save_dt=0.1, debug=True)
    np.save("test/abp.npy", out)
