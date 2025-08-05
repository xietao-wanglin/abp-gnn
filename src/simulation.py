import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
from tqdm import tqdm
from scipy.optimize import root
from time import time


class ParticleType:
    BOUNDARY = 0
    ACTIVE = 1
    PASSIVE = 2
    SIZE = 3


class Simulation:
    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray | float] = None,
        rot_rate: Optional[np.ndarray | float] = None,
        rot_couple: Optional[np.ndarray | float] = None,
        couple_radius: Optional[float] = 0.1,
        particle_type: Optional[np.ndarray] = None,
        box_length: Optional[float] = 1.0,
        timesteps: Optional[int] = 100,
        delta_t: Optional[float] = 0.1,
        periodic: Optional[bool] = True,
        solver_times: Optional[bool] = False,
        seed: Optional[int] = 0,
    ):
        np.random.seed(seed)
        if initial_state is None:
            self.n = 8
            initial_state = np.random.random(3 * self.n) * box_length
            initial_state[2::3] = initial_state[2::3] * 2 * np.pi / box_length
            initial_state = initial_state.reshape(self.n, 3).T
        else:
            self.n = initial_state.shape[1]
        self.v0 = set_param(v0, self.n, 0.1)
        self.rot_rate = set_param(rot_rate, self.n, 1)
        self.rot_couple = set_param(rot_couple, self.n, 0.1)
        self.couple_radius = couple_radius
        self.particle_type = set_param(particle_type, self.n, 1)
        self.box_length = box_length

        self.timesteps = timesteps
        self.delta_t = delta_t
        self.periodic = periodic
        self.solver_times = solver_times

        self.positions = np.zeros(shape=(self.timesteps, 3, self.n))
        self.positions[0] = initial_state
        self.pos_absolute = np.zeros(shape=(self.timesteps, 3, self.n))
        self.pos_absolute[0] = initial_state
        self.derivatives = np.zeros(shape=(self.timesteps - 1, 3, self.n))
        self.times = np.arange(0, self.delta_t * self.timesteps, self.delta_t)

    def particle_system(self, t, positions):
        positions = positions.reshape(self.n, 3).T
        x = positions[0]
        y = positions[1]
        theta = positions[2]

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
        dxdt = self.v0 * np.cos(theta) * boundary_mask
        dydt = self.v0 * np.sin(theta) * boundary_mask

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        if self.periodic:
            dx = dx - self.box_length * np.round(dx / self.box_length)
            dy = dy - self.box_length * np.round(dy / self.box_length)
        distances = np.sqrt(dx**2 + dy**2)

        neighbor_mask = (distances < self.couple_radius) & (distances > 0)

        interaction = np.sum(
            neighbor_mask * np.sin(theta[None, :] - theta[:, None]), axis=1
        )

        dthetadt = (self.rot_rate + self.rot_couple * interaction) * boundary_mask
        derivative = np.vstack([dxdt, dydt, dthetadt])
        return derivative.T.reshape(3 * self.n)

    def apply_periodic_boundary(self, positions):
        positions = positions.copy()
        positions[::3] = positions[::3] % self.box_length
        positions[1::3] = positions[1::3] % self.box_length
        positions[2::3] = positions[2::3] % (2 * np.pi)
        return positions

    def solve_dynamics(
        self,
        method: Optional[str] = "RK45",
        max_time: Optional[float] = 1,
        debug: Optional[bool] = False,
    ):
        if debug:
            start = time()
        if self.periodic:
            for i, t in enumerate(
                tqdm(self.times[:-1], leave=debug, desc="Simulation")
            ):
                derivatives = self.particle_system(
                    t, self.positions[i].T.reshape(3 * self.n)
                )
                if method == "Euler":
                    next_state = (
                        self.positions[i].T.reshape(3 * self.n)
                        + self.delta_t * derivatives
                    )
                elif method == "Backward Euler":

                    def equation(y):
                        return (
                            y
                            - self.positions[i].T.reshape(3 * self.n)
                            - self.delta_t * self.particle_system(t + self.delta_t, y)
                        )

                    initial_guess = self.positions[i].T.reshape(3 * self.n)
                    x = root(equation, initial_guess, method="lm")
                    next_state = x.x
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
                next_state = self.apply_periodic_boundary(next_state)
                self.positions[i + 1] = next_state.reshape(self.n, 3).T
                self.derivatives[i] = derivatives.reshape(self.n, 3).T

        else:
            if self.solver_times:
                sol = solve_ivp(
                    self.particle_system,
                    t_span=(0, max_time),
                    y0=self.positions[0].T.reshape(3 * self.n),
                    method=method,
                    atol=1e-9,
                    rtol=1e-6,
                )
                self.times = sol.t
            else:
                sol = solve_ivp(
                    self.particle_system,
                    t_span=(0, self.times[-1]),
                    y0=self.positions[0].T.reshape(3 * self.n),
                    t_eval=self.times,
                    method=method,
                    atol=1e-9,
                    rtol=1e-6,
                )
            self.pos_absolute = (sol.y).T.reshape(-1, self.n, 3).transpose((0, 2, 1))
            self.positions = (sol.y).T.reshape(-1, self.n, 3).transpose((0, 2, 1))
        if debug:
            end = time()
            print(f"Time elapsed: {end - start:.2f} seconds")

    def create_animation(
        self,
        timesteps: Optional[int] = None,
        filename: Optional[str] = None,
        xlim: Optional[Tuple] = None,
        ylim: Optional[Tuple] = None,
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
            ax.set_title(rf"Time: {times[fn]:2f}")
            points.set_offsets(np.c_[positions[fn][0], positions[fn][1]])
            points.set_array(positions[fn][2])

            f.canvas.draw_idle()

        if timesteps is None:
            timesteps = self.timesteps
        animation = FuncAnimation(f, update, interval=50, frames=timesteps)
        if filename is not None:
            animation.save(f"./videos/{filename}", writer="ffmpeg", fps=20)
        plt.show()

    def get_solution(self):
        return self.times, self.positions

    def get_solution_abs(self):
        return self.times, self.pos_absolute

    def get_derivatives(self):
        return self.times, self.derivatives

    def get_extended_solutions(self):
        params = np.vstack([self.rot_couple, self.v0, self.rot_couple])
        params = params[np.newaxis, ...]
        params_repeat = np.repeat(params, self.timesteps, axis=0)
        extended_solution = np.concat([self.pos_absolute, params_repeat], axis=1)
        return self.times, extended_solution


class LennardJonesSimulation(Simulation):
    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray | float] = None,
        rot_rate: Optional[np.ndarray | int] = None,
        rot_couple: Optional[np.ndarray | float] = None,
        couple_radius: Optional[float] = 0.1,
        particle_type: Optional[np.ndarray] = None,
        epsilon: Optional[float] = 0.1,
        sigma: Optional[float] = 0.01,
        box_length: Optional[float] = 1.0,
        timesteps: Optional[int] = 100,
        delta_t: Optional[float] = 0.1,
        periodic: Optional[bool] = True,
        solver_times: Optional[bool] = False,
        seed: Optional[int] = 0,
    ):
        super().__init__(
            v0=v0,
            rot_rate=rot_rate,
            rot_couple=rot_couple,
            couple_radius=couple_radius,
            particle_type=particle_type,
            box_length=box_length,
            timesteps=timesteps,
            delta_t=delta_t,
            initial_state=initial_state,
            solver_times=solver_times,
            periodic=periodic,
            seed=seed,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def repulsion(self, dx, dy, r_cutoff=np.inf):
        distances = np.sqrt(dx**2 + dy**2)

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


class RepulsiveSimulation(Simulation):
    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray | float] = None,
        rot_rate: Optional[np.ndarray | int] = None,
        rot_couple: Optional[np.ndarray | float] = None,
        particle_type: Optional[np.ndarray] = None,
        couple_radius: Optional[float] = 0.1,
        epsilon: Optional[float] = 0.1,
        sigma: Optional[float] = 0.01,
        box_length: Optional[float] = 1.0,
        timesteps: Optional[int] = 100,
        delta_t: Optional[float] = 0.1,
        periodic: Optional[bool] = True,
        solver_times: Optional[bool] = False,
        seed: Optional[int] = 0,
    ):
        super().__init__(
            v0=v0,
            rot_rate=rot_rate,
            rot_couple=rot_couple,
            couple_radius=couple_radius,
            particle_type=particle_type,
            box_length=box_length,
            timesteps=timesteps,
            delta_t=delta_t,
            initial_state=initial_state,
            solver_times=solver_times,
            periodic=periodic,
            seed=seed,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def repulsion(self, dx, dy, r_cutoff=np.inf):
        distances = np.sqrt(dx**2 + dy**2)

        mask = (distances < r_cutoff) & (distances > 0)

        inv_r = 1.0 / distances[mask]
        inv_r6 = (self.sigma * inv_r) ** 6
        inv_r12 = inv_r6**2
        F_mag = 4 * self.epsilon * (12 * inv_r12 + 6 * inv_r6) * inv_r

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


class WCASimulation(Simulation):
    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray | float] = None,
        rot_rate: Optional[np.ndarray | int] = None,
        rot_couple: Optional[np.ndarray | float] = None,
        couple_radius: Optional[float] = 0.1,
        particle_type: Optional[np.ndarray] = None,
        epsilon: Optional[float] = 0.1,
        sigma: Optional[float] = 0.01,
        box_length: Optional[float] = 1.0,
        timesteps: Optional[int] = 100,
        delta_t: Optional[float] = 0.1,
        periodic: Optional[bool] = True,
        solver_times: Optional[bool] = False,
        seed: Optional[int] = 0,
    ):
        super().__init__(
            v0=v0,
            rot_rate=rot_rate,
            rot_couple=rot_couple,
            couple_radius=couple_radius,
            particle_type=particle_type,
            box_length=box_length,
            timesteps=timesteps,
            delta_t=delta_t,
            initial_state=initial_state,
            solver_times=solver_times,
            periodic=periodic,
            seed=seed,
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


class SoftRepulsionSimulation(Simulation):
    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray | float] = None,
        rot_rate: Optional[np.ndarray | int] = None,
        rot_couple: Optional[np.ndarray | float] = None,
        couple_radius: Optional[float] = 0.1,
        particle_type: Optional[np.ndarray] = None,
        epsilon: Optional[float] = 0.1,
        sigma: Optional[float] = 0.01,
        box_length: Optional[float] = 1.0,
        timesteps: Optional[int] = 100,
        delta_t: Optional[float] = 0.1,
        periodic: Optional[bool] = True,
        solver_times: Optional[bool] = False,
        seed: Optional[int] = 0,
    ):
        super().__init__(
            v0=v0,
            rot_rate=rot_rate,
            rot_couple=rot_couple,
            couple_radius=couple_radius,
            particle_type=particle_type,
            box_length=box_length,
            timesteps=timesteps,
            delta_t=delta_t,
            initial_state=initial_state,
            solver_times=solver_times,
            periodic=periodic,
            seed=seed,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def repulsion(self, dx, dy):
        r_cutoff = np.inf

        distances = np.sqrt(dx**2 + dy**2)

        mask = (distances > 0) & (distances < r_cutoff)
        exp_factor = np.empty_like(distances)
        exp_factor.fill(0.0)
        rs = distances[mask] / self.sigma
        exp_factor[mask] = np.exp(-(rs**2))

        prefac = (2.0 * self.epsilon) / (self.sigma**2)

        Fx = prefac * exp_factor * dx
        Fy = prefac * exp_factor * dy

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


def set_param(param, n, default):
    if param is None:
        return np.ones(n) * default
    elif isinstance(param, (float, int)):
        return np.ones(n) * param
    else:
        return param
