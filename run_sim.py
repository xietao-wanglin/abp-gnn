from src.fast_simulation import WCA
import jax
import jax.numpy as jnp
import numpy as np
import math
import equinox as eqx

jax.config.update("jax_enable_x64", False)


@eqx.filter_jit
def run_sim(model, t_end, save_dt):
    return model.solve_dynamics(
        t_end=t_end, dt=1e-12, save_dt=save_dt, debug=False, use_controller=True
    )


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
    n = 32
    save_dt = 0.1
    t_end = 20
    sigma = 0.04

    particle_type, box_length, initial_state = generate_state(
        n=n, rho=0.28, sigma=sigma, rep=0, random=None
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

    out = run_sim(sim, t_end=t_end, save_dt=save_dt)
    res = np.array(out)
    np.savez(
        "sim_out.npz",
        predictions=res,
        box_length=box_length,
        initial_state=np.array(initial_state),
        dt=save_dt,
    )

    import time

    # 1. Measure Eager Execution (Standard Python loop overhead)
    # Note: This is usually very slow.
    start = time.perf_counter()
    out_eager = sim.solve_dynamics(
        t_end=1.0, dt=1e-12, save_dt=0.1, use_controller=True, debug=True
    )
    out_eager.block_until_ready()  # Wait for GPU to finish
    eager_time = time.perf_counter() - start

    # 2. Measure JIT Compilation + First Run
    # This includes the time XLA takes to optimize your code.
    @eqx.filter_jit
    def run_sim(model):
        return model.solve_dynamics(
            t_end=1.0, dt=1e-12, save_dt=0.1, use_controller=True, debug=True
        )

    start = time.perf_counter()
    out_first = run_sim(sim).block_until_ready()
    compile_and_run_time = time.perf_counter() - start

    # 3. Measure Pure Execution (The "Production" speed)
    # This is what you actually care about for long simulations.
    start = time.perf_counter()
    out_fast = run_sim(sim).block_until_ready()
    fast_time = time.perf_counter() - start

    print(f"Eager Time: {eager_time:.4f}s")
    print(f"Compile + First Run: {compile_and_run_time:.4f}s")
    print(f"Post-JIT Run: {fast_time:.4f}s")
    print(f"Speedup: {eager_time / fast_time:.1f}x")
