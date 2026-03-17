import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from time import time
from scipy.spatial import KDTree


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
        rot_couple=None,
        couple_radius=0.1,
        particle_type=None,
        box_length=1.0,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        record_every=1,
        start_record=0,
    ):
        self.initial_state = initial_state.copy()
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
        if boundary_type is None:
            self.boundary_type = (
                BoundaryType.PERIODIC,
                BoundaryType.PERIODIC,
                BoundaryType.PERIODIC,
            )
        else:
            self.boundary_type = boundary_type

        self.record_every = record_every
        self.start_record = max(0, start_record)

        total_records = max(
            0, (self.timesteps - self.start_record) // self.record_every + 1
        )
        self.positions = np.zeros(shape=(total_records, 3, self.n))
        self.pos_absolute = np.zeros(shape=(total_records, 3, self.n))
        if self.start_record == 0:
            self.positions[0] = initial_state.copy()
            self.pos_absolute[0] = initial_state.copy()

        self.times = np.arange(
            self.start_record * self.delta_t,
            self.timesteps * self.delta_t,
            self.delta_t * self.record_every,
        )

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
        atol=1e-9,
        rtol=1e-6,
    ):
        if debug:
            start = time()

        save_idx = 1
        current_state = self.initial_state.T.reshape(3 * self.n)
        pbar = tqdm(range(self.timesteps - 1), leave=debug, desc="Simulation")
        for i in pbar:
            t = i * self.delta_t
            if method == "Euler":
                derivatives = self.particle_system(t, current_state)
                next_state = current_state + self.delta_t * derivatives
            else:
                sol = solve_ivp(
                    self.particle_system,
                    t_span=(t, t + self.delta_t),
                    y0=current_state,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                )
                next_state = sol.y[:, -1]

            if debug:
                pbar.set_postfix(
                    {"avg_timestep": np.mean(np.diff(sol.t)), "n_steps": sol.t.shape[0]}
                )

            abs_next = next_state.copy()
            if self.diffusion_t > 0:
                next_state[::3] += np.sqrt(
                    2 * self.diffusion_t * self.delta_t
                ) * np.random.normal(loc=0, scale=1, size=next_state[::3].shape)
                next_state[1::3] += np.sqrt(
                    2 * self.diffusion_t * self.delta_t
                ) * np.random.normal(loc=0, scale=1, size=next_state[::3].shape)
            if self.diffusion_r > 0:
                next_state[2::3] += np.sqrt(
                    2 * self.diffusion_r * self.delta_t
                ) * np.random.normal(loc=0, scale=1, size=next_state[::3].shape)
            next_state = self.apply_periodic_boundary(next_state)

            if (
                i >= self.start_record
                and (i - self.start_record) % self.record_every == 0
            ):
                if not (i == 0 and self.start_record > 0):
                    self.positions[save_idx] = next_state.reshape(self.n, 3).T
                    self.pos_absolute[save_idx] = abs_next.reshape(self.n, 3).T
                save_idx += 1
            current_state = next_state
        if debug:
            end = time()
            print(f"Time elapsed: {end - start:.2f} seconds")

    def create_animation(
        self,
        timesteps=None,
        filename=None,
        xlim=None,
        ylim=None,
        every=None,
        trail_length=None,
        color_feature=None,
    ):
        times, positions = self.get_solution()
        if every is None:
            every = 1
        f = plt.figure(figsize=(6, 6))
        ax = f.add_subplot(111)
        ax.set_xlim(xlim if xlim is not None else (0, self.box_length))
        ax.set_ylim(ylim if ylim is not None else (0, self.box_length))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(r"Time: 0.0")
        c = color_feature if color_feature is not None else positions[0][2]
        vmin = 0 if color_feature is None else None
        vmax = 2 * np.pi if color_feature is None else None
        s = (
            200000 * (self.sigma / self.box_length) ** 2
            if self.sigma is not None
            else None
        )
        points = ax.scatter(
            positions[0][0], positions[0][1], c=c, vmin=vmin, vmax=vmax, alpha=0.6, s=s
        )

        if trail_length is not None:
            trail_segments = []
            for _particle in range(self.n):
                (line,) = ax.plot([], [], lw=2, color="red", linestyle="--", alpha=0.4)
                trail_segments.append(line)

        def update(fn):
            fn = every * fn
            ax.set_title(rf"Time: {times[fn]:2f}")
            points.set_offsets(np.c_[positions[fn][0], positions[fn][1]])
            if color_feature is None:
                points.set_array(positions[fn][2])

            if trail_length is not None:
                start_idx = max(0, fn - trail_length)
                for i in range(self.n):
                    x_trail = positions[start_idx : fn + 1, 0, i]
                    y_trail = positions[start_idx : fn + 1, 1, i]
                    trail_segments[i].set_data(x_trail, y_trail)

                return points, *trail_segments

            return points

        if timesteps is None:
            timesteps = times.shape[0] // every
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
        diffusion_t=0.0,
        diffusion_r=0.0,
        sigma=None,
        epsilon=1.0,
        gamma=0.0,
        rot_couple=None,
        couple_radius=0.0,
        particle_type=None,
        box_length=1,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        record_every=1,
        start_record=0,
    ):
        super().__init__(
            initial_state,
            v0,
            rot_rate,
            diffusion_t,
            diffusion_r,
            rot_couple,
            couple_radius,
            particle_type,
            box_length,
            timesteps,
            delta_t,
            boundary_type,
            record_every,
            start_record,
        )
        self.epsilon = epsilon
        self.sigma = set_param(sigma, self.n, 0.01)
        self.gamma = gamma

        self.pair_sigma = (self.sigma[:, None] + self.sigma[None, :]) / 2

    def repulsion(self, dx, dy, distances):
        r_cutoff = (2 ** (1 / 6)) * self.pair_sigma

        mask = (distances < r_cutoff) & (distances > 0)

        inv_r = 1.0 / distances[mask]
        inv_r6 = (self.pair_sigma[mask] * inv_r) ** 6
        inv_r12 = inv_r6**2
        F_mag = 24 * self.epsilon * (2 * inv_r12 - inv_r6) * inv_r

        Fx = np.zeros_like(distances)
        Fy = np.zeros_like(distances)
        Fx[mask] = F_mag * dx[mask] / distances[mask]
        Fy[mask] = F_mag * dy[mask] / distances[mask]

        Fx_total = np.sum(Fx, axis=1)
        Fy_total = np.sum(Fy, axis=1)

        return Fx_total, Fy_total

    def particle_system(self, t, positions):
        positions = positions.reshape(self.n, 3).T
        x = positions[0]
        y = positions[1]
        theta = positions[2]

        if self.boundary_type[0] == BoundaryType.PERIODIC:
            x = x % self.box_length
        if self.boundary_type[1] == BoundaryType.PERIODIC:
            y = y % self.box_length
        if self.boundary_type[2] == BoundaryType.PERIODIC:
            theta = theta % (2 * np.pi)

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

        Fx_total, Fy_total = self.repulsion(dx, dy, distances)

        steric_torque = np.cos(theta) * Fy_total - np.sin(theta) * Fx_total

        dxdt += Fx_total
        dydt += Fy_total

        dthetadt = (
            self.rot_rate + self.rot_couple * interaction + self.gamma * steric_torque
        )
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
        record_every=1,
        start_record=0,
    ):
        super().__init__(
            initial_state,
            v0,
            rot_rate,
            diffusion_t,
            diffusion_r,
            rot_couple,
            couple_radius,
            particle_type,
            box_length,
            timesteps,
            delta_t,
            boundary_type,
            record_every,
            start_record,
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
        epsilon=1,
        rot_couple=None,
        couple_radius=0.1,
        particle_type=None,
        box_length=1,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        record_every=1,
        start_record=0,
    ):
        super().__init__(
            initial_state,
            v0,
            rot_rate,
            diffusion_t,
            diffusion_r,
            rot_couple,
            couple_radius,
            particle_type,
            box_length,
            timesteps,
            delta_t,
            boundary_type,
            record_every,
            start_record,
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


class SparseWCA(BaseSimulation):
    def __init__(
        self,
        initial_state,
        v0=None,
        rot_rate=None,
        diffusion_t=0.0,
        diffusion_r=0.0,
        sigma=0.01,
        epsilon=1.0,
        rot_couple=None,
        couple_radius=0.0,
        particle_type=None,
        box_length=1,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        record_every=1,
        start_record=0,
    ):
        super().__init__(
            initial_state,
            v0,
            rot_rate,
            diffusion_t,
            diffusion_r,
            rot_couple,
            couple_radius,
            particle_type,
            box_length,
            timesteps,
            delta_t,
            boundary_type,
            record_every,
            start_record,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def particle_system(self, t, positions):
        positions = positions.reshape(self.n, 3).T
        x, y, theta = positions[0], positions[1], positions[2]

        if self.boundary_type[0] == BoundaryType.PERIODIC:
            x = x % self.box_length
        if self.boundary_type[1] == BoundaryType.PERIODIC:
            y = y % self.box_length
        if self.boundary_type[2] == BoundaryType.PERIODIC:
            theta = theta % (2 * np.pi)

        dxdt = self.v0 * np.cos(theta)
        dydt = self.v0 * np.sin(theta)

        Fx_total = np.zeros(self.n)
        Fy_total = np.zeros(self.n)
        interaction = np.zeros(self.n)

        pos_2d = np.stack([x, y], axis=1)

        r_cutoff_wca = (2 ** (1 / 6)) * self.sigma
        r_max = max(r_cutoff_wca, self.couple_radius)
        tree = KDTree(pos_2d, boxsize=[self.box_length, self.box_length])
        pairs = tree.query_pairs(r=r_max, output_type="ndarray")

        if pairs.size > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]

            pos_i = pos_2d[i]
            pos_j = pos_2d[j]

            dx = pos_i[:, 0] - pos_j[:, 0]
            dy = pos_i[:, 1] - pos_j[:, 1]
            dx = dx - self.box_length * np.round(dx / self.box_length)
            dy = dy - self.box_length * np.round(dy / self.box_length)

            distances = np.sqrt(dx**2 + dy**2)
            mask_wca = (distances < r_cutoff_wca) & (distances > 0)

            if np.any(mask_wca):
                dist_wca = distances[mask_wca]
                dx_wca = dx[mask_wca]
                dy_wca = dy[mask_wca]

                inv_r = 1.0 / dist_wca
                inv_r6 = (self.sigma * inv_r) ** 6
                inv_r12 = inv_r6**2
                F_mag = 24 * self.epsilon * (2 * inv_r12 - inv_r6) * inv_r

                Fx = F_mag * dx_wca / dist_wca
                Fy = F_mag * dy_wca / dist_wca

                i_wca = i[mask_wca]
                j_wca = j[mask_wca]

                np.add.at(Fx_total, i_wca, Fx)
                np.add.at(Fx_total, j_wca, -Fx)
                np.add.at(Fy_total, i_wca, Fy)
                np.add.at(Fy_total, j_wca, -Fy)

            mask_couple = (distances < self.couple_radius) & (distances > 0)

            if np.any(mask_couple):
                i_couple = i[mask_couple]
                j_couple = j[mask_couple]

                theta_i = theta[i_couple]
                theta_j = theta[j_couple]

                sin_dtheta = np.sin(theta_j - theta_i)
                np.add.at(interaction, i_couple, sin_dtheta)
                np.add.at(interaction, j_couple, -sin_dtheta)

        dxdt += Fx_total
        dydt += Fy_total
        dthetadt = self.rot_rate + self.rot_couple * interaction

        boundary_mask = self.particle_type > ParticleType.BOUNDARY
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
