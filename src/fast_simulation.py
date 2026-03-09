import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx


class ParticleType:
    BOUNDARY = 0
    ACTIVE = 1
    SIZE = 2


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

    def solve_dynamics(
        self,
        t_end,
        wrap=None,
        dt=None,
        save_dt=None,
        min_save_t=None,
        use_controller=None,
        debug=None,
    ):
        if wrap is None:
            wrap = True
        if dt is None:
            dt = 1e-4
        if save_dt is None:
            save_dt = dt
        if min_save_t is None:
            min_save_t = 0

        progress_bar = dfx.TqdmProgressMeter() if debug else dfx.NoProgressMeter()

        stepsize_controller = (
            dfx.PIDController(rtol=1e-6, atol=1e-9)
            if use_controller
            else dfx.ConstantStepSize()
        )
        times = jnp.arange(min_save_t, t_end, save_dt)
        solver = dfx.Heun()

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
            stepsize_controller=stepsize_controller,
            max_steps=None,
            progress_meter=progress_bar,
        )

        y_hist = sol.ys
        pos = y_hist[:, :2, :]
        theta = y_hist[:, 2, :]

        if wrap:
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
        fx = -self.epsilon * dx
        fy = -self.epsilon * dy

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

    r_cutoff_sq: float
    couple_radius_sq: float
    force_prefactor: float
    sigma_sq: float

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

        self.r_cutoff_sq = ((2 ** (1 / 6)) * sigma) ** 2
        self.couple_radius_sq = couple_radius**2
        self.force_prefactor = 24 * epsilon
        self.sigma_sq = sigma**2

    def particle_system(self, t, y, args):
        pos = y[:2].T
        theta = y[2]

        diff = pos[:, None, :] - pos[None, :, :]
        diff = diff - self.box_length * jnp.rint(diff / self.box_length)
        dist_sq = jnp.sum(diff**2, axis=-1)

        wca_mask = (dist_sq < self.r_cutoff_sq) & (dist_sq > 0.0)
        safe_dist_sq = jnp.where(wca_mask, dist_sq, 1.0)
        inv_r2 = 1.0 / safe_dist_sq
        sigma_r6 = (self.sigma_sq * inv_r2) ** 3
        f_term = (
            self.force_prefactor * (2 * sigma_r6**2 - sigma_r6) * inv_r2
        ) * wca_mask

        fx_total = jnp.sum(f_term * diff[..., 0], axis=1)
        fy_total = jnp.sum(f_term * diff[..., 1], axis=1)

        angle_diff = jnp.sin(theta[None, :] - theta[:, None])
        align_mask = (dist_sq < self.couple_radius_sq) & (dist_sq > 0.0)
        alignment = jnp.sum(angle_diff * align_mask, axis=1)

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = (self.v0 * jnp.cos(theta) + fx_total) * boundary_mask
        dydt = (self.v0 * jnp.sin(theta) + fy_total) * boundary_mask
        dthetadt = (self.rot_rate + self.couple_strength * alignment) * boundary_mask

        return jnp.stack([dxdt, dydt, dthetadt])
    
class BoundaryWCA(FastSimulation):
    v0: float
    rot_rate: float
    epsilon: float
    sigma: float
    couple_radius: float
    couple_strength: float
    particle_type: jax.Array

    r_cutoff_sq: float
    couple_radius_sq: float
    force_prefactor: float
    sigma_sq: float

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

        self.r_cutoff_sq = ((2 ** (1 / 6)) * sigma) ** 2
        self.couple_radius_sq = couple_radius**2
        self.force_prefactor = 24 * epsilon
        self.sigma_sq = sigma**2

    def particle_system(self, t, y, args):
        pos = y[:2].T
        theta = y[2]

        diff = pos[:, None, :] - pos[None, :, :]
        diff = diff - self.box_length * jnp.rint(diff / self.box_length)
        dist_sq = jnp.sum(diff**2, axis=-1)

        is_boundary = (self.particle_type == ParticleType.BOUNDARY)
        type_mask = is_boundary[:, None] | is_boundary[None, :]
        wca_mask = (dist_sq < self.r_cutoff_sq) & (dist_sq > 0.0) & type_mask
        
        safe_dist_sq = jnp.where(wca_mask, dist_sq, 1.0)
        inv_r2 = 1.0 / safe_dist_sq
        sigma_r6 = (self.sigma_sq * inv_r2) ** 3
        f_term = (
            self.force_prefactor * (2 * sigma_r6**2 - sigma_r6) * inv_r2
        ) * wca_mask

        fx_total = jnp.sum(f_term * diff[..., 0], axis=1)
        fy_total = jnp.sum(f_term * diff[..., 1], axis=1)

        angle_diff = jnp.sin(theta[None, :] - theta[:, None])
        align_mask = (dist_sq < self.couple_radius_sq) & (dist_sq > 0.0) & type_mask
        alignment = jnp.sum(angle_diff * align_mask, axis=1)

        movement_mask = self.particle_type > ParticleType.BOUNDARY
        
        dxdt = (self.v0 * jnp.cos(theta) + fx_total) * movement_mask
        dydt = (self.v0 * jnp.sin(theta) + fy_total) * movement_mask
        dthetadt = (self.rot_rate + self.couple_strength * alignment) * movement_mask

        return jnp.stack([dxdt, dydt, dthetadt])


class OscillationWCA(FastSimulation):
    v0: float
    rot_rate: float
    osc_amp: float
    osc_freq: float
    epsilon: float
    sigma: float
    couple_radius: float
    couple_strength: float
    particle_type: jax.Array

    r_cutoff_sq: float
    couple_radius_sq: float
    force_prefactor: float
    sigma_sq: float

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
        osc_amp,
        osc_freq,
        box_length=1.0,
    ):
        super().__init__(initial_state, box_length)
        self.v0 = v0
        self.rot_rate = rot_rate
        self.osc_amp = osc_amp
        self.osc_freq = osc_freq
        self.epsilon = epsilon
        self.sigma = sigma
        self.couple_radius = couple_radius
        self.couple_strength = couple_strength
        self.particle_type = particle_type

        self.r_cutoff_sq = ((2 ** (1 / 6)) * sigma) ** 2
        self.couple_radius_sq = couple_radius**2
        self.force_prefactor = 24 * epsilon
        self.sigma_sq = sigma**2

    def particle_system(self, t, y, args):
        pos = y[:2].T
        theta = y[2]

        diff = pos[:, None, :] - pos[None, :, :]
        diff = diff - self.box_length * jnp.rint(diff / self.box_length)
        dist_sq = jnp.sum(diff**2, axis=-1)

        wca_mask = (dist_sq < self.r_cutoff_sq) & (dist_sq > 0.0)
        safe_dist_sq = jnp.where(wca_mask, dist_sq, 1.0)
        inv_r2 = 1.0 / safe_dist_sq
        sigma_r6 = (self.sigma_sq * inv_r2) ** 3
        f_term = (
            self.force_prefactor * (2 * sigma_r6**2 - sigma_r6) * inv_r2
        ) * wca_mask

        fx_total = jnp.sum(f_term * diff[..., 0], axis=1)
        fy_total = jnp.sum(f_term * diff[..., 1], axis=1)

        angle_diff = jnp.sin(theta[None, :] - theta[:, None])
        align_mask = (dist_sq < self.couple_radius_sq) & (dist_sq > 0.0)
        alignment = jnp.sum(angle_diff * align_mask, axis=1)

        oscillation = self.osc_amp * jnp.sin(self.osc_freq * t)

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = (self.v0 * jnp.cos(theta) + fx_total) * boundary_mask
        dydt = (self.v0 * jnp.sin(theta) + fy_total) * boundary_mask
        
        dthetadt = (self.rot_rate + oscillation + self.couple_strength * alignment) * boundary_mask

        return jnp.stack([dxdt, dydt, dthetadt])
    

class TorqueWCA(FastSimulation):
    v0: float
    rot_rate: float
    gamma: float
    epsilon: float
    sigma: float
    couple_radius: float
    couple_strength: float
    particle_type: jax.Array

    r_cutoff_sq: float
    couple_radius_sq: float
    force_prefactor: float
    sigma_sq: float

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
        gamma,
        box_length=1.0,
    ):
        super().__init__(initial_state, box_length)
        self.v0 = v0
        self.rot_rate = rot_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma = sigma
        self.couple_radius = couple_radius
        self.couple_strength = couple_strength
        self.particle_type = particle_type

        self.r_cutoff_sq = ((2 ** (1 / 6)) * sigma) ** 2
        self.couple_radius_sq = couple_radius**2
        self.force_prefactor = 24 * epsilon
        self.sigma_sq = sigma**2

    def particle_system(self, t, y, args):
        pos = y[:2].T
        theta = y[2]

        diff = pos[:, None, :] - pos[None, :, :]
        diff = diff - self.box_length * jnp.rint(diff / self.box_length)
        dist_sq = jnp.sum(diff**2, axis=-1)

        wca_mask = (dist_sq < self.r_cutoff_sq) & (dist_sq > 0.0)
        safe_dist_sq = jnp.where(wca_mask, dist_sq, 1.0)
        inv_r2 = 1.0 / safe_dist_sq
        sigma_r6 = (self.sigma_sq * inv_r2) ** 3
        f_term = (
            self.force_prefactor * (2 * sigma_r6**2 - sigma_r6) * inv_r2
        ) * wca_mask

        fx_total = jnp.sum(f_term * diff[..., 0], axis=1)
        fy_total = jnp.sum(f_term * diff[..., 1], axis=1)

        steric_torque = (jnp.cos(theta) * fy_total - jnp.sin(theta) * fx_total)
        angle_diff = jnp.sin(theta[None, :] - theta[:, None])
        align_mask = (dist_sq < self.couple_radius_sq) & (dist_sq > 0.0)
        alignment = jnp.sum(angle_diff * align_mask, axis=1)


        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = (self.v0 * jnp.cos(theta) + fx_total) * boundary_mask
        dydt = (self.v0 * jnp.sin(theta) + fy_total) * boundary_mask
        
        dthetadt = (self.rot_rate + self.gamma * steric_torque + self.couple_strength * alignment) * boundary_mask

        return jnp.stack([dxdt, dydt, dthetadt])
