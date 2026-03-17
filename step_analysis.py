from src.fast_simulation import WCA
import jax
import jax.numpy as jnp
import numpy as np
import math
import equinox as eqx

jax.config.update("jax_enable_x64", True)


@eqx.filter_jit
def run_sim(model, dt, t_end, save_dt):
    return model.solve_dynamics(
        t_end=t_end, dt=dt, save_dt=save_dt, debug=True, use_controller=False
    )


def load_state():
    n = 64
    rho = 0.28
    sigma = 0.04
    data = np.load("step/gt.npy")[0]
    initial_state = jnp.asarray(data)
    box_length = sigma * math.sqrt(n * math.pi / rho) / 2

    particle_type = jnp.ones(n)

    return particle_type, box_length, initial_state


if __name__ == "__main__":
    start = -6
    end = -1
    n = 41

    exps = [start + i * (end - start) / (n - 1) for i in range(n)]

    for i, exp in enumerate(exps):
        sim_dt = 10 ** (exp)
        save_dt = 1
        t_end = 20
        sigma = 0.04

        particle_type, box_length, initial_state = load_state()
        sim = WCA(
            initial_state=initial_state,
            v0=0.1,
            rot_rate=0.0,
            epsilon=0.1,
            sigma=sigma,
            couple_radius=0.0,
            couple_strength=0.0,
            particle_type=particle_type,
            box_length=box_length,
        )

        out = run_sim(sim, dt=sim_dt, t_end=t_end, save_dt=save_dt)
        res = np.array(out)
        np.savez(
            f"step/sim_{i}.npz",
            predictions=res,
            box_length=box_length,
            initial_state=np.array(initial_state),
            dt=save_dt,
            sim_dt=sim_dt,
        )
