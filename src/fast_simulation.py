import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx


class ParticleType:
    BOUNDARY = 0
    ACTIVE = 1
    PASSIVE = 2
    SIZE = 3


class FastSimulation(eqx.Module):
    initial_state: jax.Array
    box_length: float
    n: int

    def __init__(self, initial_state, box_length=1.0):
        self.initial_state = initial_state
        self.n = initial_state.shape[1]
        self.box_length = box_length

    def particle_system(self, t, y, args):
        raise NotImplementedError

    @eqx.filter_jit
    def solve_dynamics(self, t_end, dt=None, save_dt=None, min_save_t=None, debug=None):
        if dt is None:
            dt = 1e-4
        if save_dt is None:
            save_dt = dt
        if min_save_t is None:
            min_save_t = 0
        if (debug is None) or (not debug):
            progress_bar = dfx.NoProgressMeter()
        if debug:
            progress_bar = dfx.TqdmProgressMeter()
        times = jnp.arange(min_save_t, t_end, save_dt)
        solver = dfx.Dopri5()
        y0 = self.initial_state
        saveat = dfx.SaveAt(ts=times)
        term = dfx.ODETerm(self.particle_system)
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0,
            t1=t_end,
            dt0=dt,
            y0=y0,
            saveat=saveat,
            max_steps=10_000_000,
            progress_meter=progress_bar,
        )

        y_hist = sol.ys
        pos = y_hist[:, :2, :]
        theta = y_hist[:, 2, :]

        pos = pos % self.box_length
        theta = theta % (2 * jnp.pi)

        y_final = jnp.concatenate([pos, theta[:, None, :]], axis=1)

        return y_final


class Toy(FastSimulation):
    v0: float
    rot_rate: float
    epsilon: float

    def __init__(self, initial_state, v0, rot_rate, epsilon, box_length=1.0):
        super().__init__(initial_state, box_length)
        self.v0 = v0
        self.rot_rate = rot_rate
        self.epsilon = epsilon

    def potential(self, dx, dy):
        f_mag = -self.epsilon
        fx = f_mag * dx
        fy = f_mag * dy

        fx_total = jnp.sum(fx, axis=1)
        fy_total = jnp.sum(fy, axis=1)

        return fx_total, fy_total

    def particle_system(self, t, y, args):
        _x = y[0]
        _y = y[1]
        _theta = y[2]

        dxdt = self.v0 * jnp.cos(_theta)
        dydt = self.v0 * jnp.sin(_theta)

        dx = _x[:, None] - _x[None, :]
        dy = _y[:, None] - _y[None, :]
        dx = dx - self.box_length * jnp.round(dx / self.box_length)
        dy = dy - self.box_length * jnp.round(dy / self.box_length)

        fx_total, fy_total = self.potential(dx, dy)
        dxdt += fx_total
        dydt += fy_total

        dthetadt = jnp.full_like(_theta, self.rot_rate)
        derivative = jnp.vstack([dxdt, dydt, dthetadt])
        return derivative


class WCA(FastSimulation):
    v0: float
    rot_rate: float
    epsilon: float
    sigma: float
    couple_radius: float
    couple_strength: float
    particle_type: jax.Array

    def __init__(
        self,
        initial_state,
        v0,
        rot_rate,
        epsilon,
        sigma,
        couple_radius,
        couple_strength,
        particle_type,
        box_length=1.0,
    ):
        super().__init__(initial_state, box_length)
        self.v0 = v0
        self.rot_rate = rot_rate
        self.epsilon = epsilon
        self.sigma = sigma
        self.couple_radius = couple_radius
        self.couple_strength = couple_strength
        self.particle_type = particle_type

    def potential(self, dx, dy, theta):
        distances = jnp.sqrt(dx**2 + dy**2)
        r_cutoff = (2 ** (1 / 6)) * self.sigma

        safe_dist = jnp.where(distances == 0.0, 1.0, distances)

        inv_r = 1.0 / safe_dist
        inv_r6 = (self.sigma * inv_r) ** 6
        inv_r12 = inv_r6**2
        f_mag = 24 * self.epsilon * (2 * inv_r12 - inv_r6) * inv_r
        mask = (distances < r_cutoff) & (distances > 0.0)
        f_mag = f_mag * mask
        fx = f_mag * dx / safe_dist
        fy = f_mag * dy / safe_dist

        fx_total = jnp.sum(fx, axis=1)
        fy_total = jnp.sum(fy, axis=1)

        neighbor_mask = (distances < self.couple_radius) & (distances > 0)

        alignment = jnp.sum(
            neighbor_mask * jnp.sin(theta[None, :] - theta[:, None]), axis=1
        )

        return fx_total, fy_total, alignment

    def particle_system(self, t, y, args):
        _x = y[0]
        _y = y[1]
        _theta = y[2]

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = self.v0 * jnp.cos(_theta)
        dydt = self.v0 * jnp.sin(_theta)

        dx = _x[:, None] - _x[None, :]
        dy = _y[:, None] - _y[None, :]

        dx = dx - self.box_length * jnp.round(dx / self.box_length)
        dy = dy - self.box_length * jnp.round(dy / self.box_length)

        fx_total, fy_total, alignment = self.potential(dx, dy, _theta)

        dxdt += fx_total
        dydt += fy_total

        dthetadt = (
            jnp.full_like(_theta, self.rot_rate) + self.couple_strength * alignment
        )

        dxdt *= boundary_mask
        dydt *= boundary_mask
        dthetadt *= boundary_mask

        derivative = jnp.vstack([dxdt, dydt, dthetadt])
        return derivative
