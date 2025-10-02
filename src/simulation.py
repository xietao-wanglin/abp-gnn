import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from time import time


class ParticleType:
    BOUNDARY = 0
    ACTIVE = 1
    PASSIVE = 2
    SIZE = 3


class BoundaryType:
    NO_BOUNDARY = 0
    PERIODIC = 1
    HARD = 2  # TODO


class BaseSimulation:
    def __init__(
        self,
        initial_state,
        v0=None,
        rot_rate=None,
        diffusion_t=0.0,
        diffusion_r=0.0,
        motility=1,
        rot_couple=None,
        couple_radius=0.1,
        particle_type=None,
        box_length=1.0,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        seed=0,
    ):
        np.random.seed(seed)
        self.n = initial_state.shape[1]
        self.v0 = set_param(v0, self.n, 0.1)
        self.rot_rate = set_param(rot_rate, self.n, 1)
        self.rot_couple = set_param(rot_couple, self.n, 0.1)
        self.couple_radius = couple_radius
        self.particle_type = set_param(particle_type, self.n, 1)
        self.box_length = box_length

        self.timesteps = timesteps
        self.delta_t = delta_t
        self.diffusion_t = diffusion_t
        self.diffusion_r = diffusion_r
        self.motility = motility
        if boundary_type is None:
            self.boundary_type = (
                BoundaryType.PERIODIC,
                BoundaryType.PERIODIC,
                BoundaryType.PERIODIC,
            )
        else:
            self.boundary_type = boundary_type

        self.positions = np.zeros(shape=(self.timesteps, 3, self.n))
        self.positions[0] = initial_state
        self.pos_absolute = np.zeros(shape=(self.timesteps, 3, self.n))
        self.pos_absolute[0] = initial_state
        self.times = np.arange(0, self.delta_t * self.timesteps, self.delta_t)

    def particle_system(self, t, positions):
        raise NotImplementedError

    def apply_periodic_boundary(self, positions):
        positions = positions.copy()
        if self.boundary_type[0] == BoundaryType.PERIODIC:
            positions[::3] = positions[::3] % self.box_length
        if self.boundary_type[1] == BoundaryType.PERIODIC:
            positions[1::3] = positions[1::3] % self.box_length
        if self.boundary_type[2] == BoundaryType.PERIODIC:
            positions[2::3] = positions[2::3] % (2 * np.pi)
        return positions

    def solve_dynamics(
        self,
        method="RK45",
        debug=False,
    ):
        if debug:
            start = time()
        for i, t in enumerate(tqdm(self.times[:-1], leave=debug, desc="Simulation")):
            if method == "Euler":
                derivatives = self.particle_system(
                    t, self.positions[i].T.reshape(3 * self.n)
                )
                next_state = (
                    self.positions[i].T.reshape(3 * self.n) + self.delta_t * derivatives
                )
            else:
                sol = solve_ivp(
                    self.particle_system,
                    t_span=(t, t + self.delta_t),
                    y0=self.positions[i].T.reshape(3 * self.n),
                    t_eval=[t + self.delta_t],
                    method=method,
                    atol=1e-9,
                    rtol=1e-6,
                )
                next_state = sol.y[:, -1]
            self.pos_absolute[i + 1] = next_state.reshape(self.n, 3).T
            next_state[::3] += np.sqrt(
                2 * self.diffusion_t * self.delta_t
            ) * np.random.normal(loc=0, scale=1, size=next_state[::3].shape)
            next_state[1::3] += np.sqrt(
                2 * self.diffusion_t * self.delta_t
            ) * np.random.normal(loc=0, scale=1, size=next_state[::3].shape)
            next_state[2::3] += np.sqrt(
                2 * self.diffusion_r * self.delta_t
            ) * np.random.normal(loc=0, scale=1, size=next_state[::3].shape)
            next_state = self.apply_periodic_boundary(next_state)
            self.positions[i + 1] = next_state.reshape(self.n, 3).T
        if debug:
            end = time()
            print(f"Time elapsed: {end - start:.2f} seconds")

    def create_animation(
        self,
        timesteps=None,
        filename=None,
        xlim=None,
        ylim=None,
        every=1,
    ):
        times, positions = self.get_solution()
        f = plt.figure(figsize=(6, 5))
        ax = f.add_subplot(111)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        else:
            ax.set_xlim(0, self.box_length)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            ax.set_ylim(0, self.box_length)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(r"Time: 0.0")
        points = ax.scatter(
            positions[0][0],
            positions[0][1],
            c=positions[0][2],
            vmin=0,
            vmax=2 * np.pi,
            alpha=0.3,
        )
        f.colorbar(points, label=r"Orientation $\theta_i$")

        def update(fn):
            fn = every * fn
            ax.set_title(rf"Time: {times[fn]:2f}")
            points.set_offsets(np.c_[positions[fn][0], positions[fn][1]])
            points.set_array(positions[fn][2])

            f.canvas.draw_idle()

        if timesteps is None:
            timesteps = int(self.timesteps / every)
        animation = FuncAnimation(f, update, interval=50, frames=timesteps)
        if filename is not None:
            animation.save(f"./videos/{filename}", writer="ffmpeg", fps=20)
        plt.show()

    def get_solution(self, every=1):
        return self.times[::every], self.positions[::every]

    def get_solution_abs(self, every=1):
        return self.times[::every], self.pos_absolute[::every]


class WCA(BaseSimulation):
    def __init__(
        self,
        initial_state,
        v0=None,
        rot_rate=None,
        diffusion_t=0.01,
        diffusion_r=0.1,
        motility=1,
        sigma=0.01,
        epsilon=0.1,
        rot_couple=None,
        couple_radius=0.1,
        particle_type=None,
        box_length=1,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        seed=0,
    ):
        super().__init__(
            initial_state,
            v0,
            rot_rate,
            diffusion_t,
            diffusion_r,
            motility,
            rot_couple,
            couple_radius,
            particle_type,
            box_length,
            timesteps,
            delta_t,
            boundary_type,
            seed,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def repulsion(self, dx, dy):
        distances = np.sqrt(dx**2 + dy**2)

        r_cutoff = (2 ** (1 / 6)) * self.sigma

        mask = (distances < r_cutoff) & (distances > 0)

        inv_r = 1.0 / distances[mask]
        inv_r6 = (self.sigma * inv_r) ** 6
        inv_r12 = inv_r6**2
        F_mag = 4 * self.epsilon * (12 * inv_r12 - 6 * inv_r6) * inv_r

        Fx = np.zeros_like(distances)
        Fy = np.zeros_like(distances)
        Fx[mask] = F_mag * dx[mask]
        Fy[mask] = F_mag * dy[mask]

        Fx_total = np.sum(Fx, axis=1)
        Fy_total = np.sum(Fy, axis=1)

        return Fx_total, Fy_total

    def particle_system(self, t, positions):
        positions = positions.reshape(self.n, 3).T
        x = positions[0]
        y = positions[1]
        theta = positions[2]

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = self.v0 * np.cos(theta)
        dydt = self.v0 * np.sin(theta)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]

        dx = dx - self.box_length * np.round(dx / self.box_length)
        dy = dy - self.box_length * np.round(dy / self.box_length)
        distances = np.sqrt(dx**2 + dy**2)

        neighbor_mask = (distances < self.couple_radius) & (distances > 0)

        interaction = np.sum(
            neighbor_mask * np.sin(theta[None, :] - theta[:, None]), axis=1
        )

        Fx_total, Fy_total = self.repulsion(dx, dy)

        dxdt += Fx_total
        dydt += Fy_total

        dthetadt = self.rot_rate + self.rot_couple * interaction
        dxdt *= boundary_mask
        dydt *= boundary_mask
        dthetadt *= boundary_mask
        derivative = np.vstack([dxdt, dydt, dthetadt])
        return derivative.T.reshape(3 * self.n)


class LennardJones(BaseSimulation):
    def __init__(
        self,
        initial_state,
        v0=None,
        rot_rate=None,
        diffusion_t=0.0,
        diffusion_r=0.0,
        motility=1,
        sigma=0.01,
        epsilon=0.1,
        law_power=6.0,
        regularisation=0.0,
        repul_strength=12.0,
        attr_strength=6.0,
        rot_couple=None,
        couple_radius=0.1,
        particle_type=None,
        box_length=1,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        seed=0,
    ):
        super().__init__(
            initial_state,
            v0,
            rot_rate,
            diffusion_t,
            diffusion_r,
            motility,
            rot_couple,
            couple_radius,
            particle_type,
            box_length,
            timesteps,
            delta_t,
            boundary_type,
            seed,
        )
        self.epsilon = epsilon
        self.sigma = sigma
        self.law_power = law_power
        self.regularisation = regularisation
        self.repul_strength = repul_strength
        self.attr_strength = attr_strength

    def repulsion(self, dx, dy, r_cutoff=np.inf):
        distances = np.sqrt(dx**2 + dy**2)

        mask = (distances < r_cutoff) & (distances > 0)

        inv_r = 1.0 / (distances[mask] + self.regularisation)
        inv_r6 = (self.sigma * inv_r) ** self.law_power
        inv_r12 = inv_r6**2
        F_mag = (
            4
            * self.epsilon
            * (self.repul_strength * inv_r12 - self.attr_strength * inv_r6)
            * inv_r
        )

        Fx = np.zeros_like(distances)
        Fy = np.zeros_like(distances)
        Fx[mask] = F_mag * dx[mask]
        Fy[mask] = F_mag * dy[mask]

        Fx_total = np.sum(Fx, axis=1)
        Fy_total = np.sum(Fy, axis=1)

        return Fx_total, Fy_total

    def particle_system(self, t, positions):
        positions = positions.reshape(self.n, 3).T
        x = positions[0]
        y = positions[1]
        theta = positions[2]

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = self.v0 * np.cos(theta)
        dydt = self.v0 * np.sin(theta)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]

        if self.boundary_type[0] == BoundaryType.PERIODIC:
            dx = dx - self.box_length * np.round(dx / self.box_length)
        if self.boundary_type[1] == BoundaryType.PERIODIC:
            dy = dy - self.box_length * np.round(dy / self.box_length)
        distances = np.sqrt(dx**2 + dy**2)

        neighbor_mask = (distances < self.couple_radius) & (distances > 0)

        interaction = np.sum(
            neighbor_mask * np.sin(theta[None, :] - theta[:, None]), axis=1
        )

        Fx_total, Fy_total = self.repulsion(dx, dy)

        dxdt += Fx_total
        dydt += Fy_total

        dthetadt = self.rot_rate + self.rot_couple * interaction
        dxdt *= boundary_mask
        dydt *= boundary_mask
        dthetadt *= boundary_mask
        derivative = np.vstack([dxdt, dydt, dthetadt])
        return derivative.T.reshape(3 * self.n)


class Toy(BaseSimulation):
    def __init__(
        self,
        initial_state,
        v0=None,
        rot_rate=None,
        diffusion_t=0.0,
        diffusion_r=0.0,
        motility=1,
        epsilon=1,
        rot_couple=None,
        couple_radius=0.1,
        particle_type=None,
        box_length=1,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        seed=0,
    ):
        super().__init__(
            initial_state,
            v0,
            rot_rate,
            diffusion_t,
            diffusion_r,
            motility,
            rot_couple,
            couple_radius,
            particle_type,
            box_length,
            timesteps,
            delta_t,
            boundary_type,
            seed,
        )
        self.epsilon = epsilon

    def repulsion(self, dx, dy):
        F_mag = -self.epsilon
        Fx = F_mag * dx
        Fy = F_mag * dy

        Fx_total = np.sum(Fx, axis=1)
        Fy_total = np.sum(Fy, axis=1)

        return Fx_total, Fy_total

    def particle_system(self, t, positions):
        positions = positions.reshape(self.n, 3).T
        x = positions[0]
        y = positions[1]
        theta = positions[2]

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = self.v0 * np.cos(theta)
        dydt = self.v0 * np.sin(theta)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]

        if self.boundary_type[0] == BoundaryType.PERIODIC:
            dx = dx - self.box_length * np.round(dx / self.box_length)
        if self.boundary_type[1] == BoundaryType.PERIODIC:
            dy = dy - self.box_length * np.round(dy / self.box_length)
        distances = np.sqrt(dx**2 + dy**2)

        neighbor_mask = (distances < self.couple_radius) & (distances > 0)

        interaction = np.sum(
            neighbor_mask * np.sin(theta[None, :] - theta[:, None]), axis=1
        )

        Fx_total, Fy_total = self.repulsion(dx, dy)

        dxdt += Fx_total
        dydt += Fy_total

        dthetadt = self.rot_rate + self.rot_couple * interaction
        dxdt *= boundary_mask
        dydt *= boundary_mask
        dthetadt *= boundary_mask
        derivative = np.vstack([dxdt, dydt, dthetadt])
        return derivative.T.reshape(3 * self.n)


def set_param(param, n, default):
    if param is None:
        return np.ones(n) * default
    elif isinstance(param, (float, int)):
        return np.ones(n) * param
    else:
        return param
