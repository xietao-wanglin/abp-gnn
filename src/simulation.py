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
        s = 200000*self.sigma**2 if self.sigma is not None else None
        points = ax.scatter(
            positions[0][0],
            positions[0][1],
            c=c,
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,
            s=s
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
        diffusion_t=0.0,
        diffusion_r=0.0,
        sigma=None,
        epsilon=1.0,
        rot_couple=None,
        couple_radius=0.0,
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
        self.sigma = set_param(sigma, self.n, 0.01)

        self.pair_sigma = (self.sigma[:, None] + self.sigma[None, :])/2

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


class AVM(BaseSimulation):
    """
    Anticipation Velocity Model (AVM)
    with collision-free speed control.

    Reference:
        Xu, Q., Chraibi, M., Seyfried, A. (2021)
        "Anticipation in a velocity-based model for pedestrian dynamics"
        Transportation Research Part C, 133 (2021) 103464
    """

    def __init__(
        self,
        initial_state,
        v0=None,  # free speed (m/s)
        tau=0.3,  # relaxation time (s)
        T=1.06,  # time gap for speed-headway relation (s)
        D=0.1,  # range parameter (m)
        k=3.0,  # alignment/following strength
        t_a=1.0,  # anticipation time (s)
        r=0.18,  # agent radius (m)
        diffusion_t=0.0,
        diffusion_r=0.0,
        particle_type=None,
        box_length=1.0,
        timesteps=100,
        delta_t=0.1,
        boundary_type=None,
        seed=0,
    ):
        super().__init__(
            initial_state,
            v0=v0,
            diffusion_t=diffusion_t,
            diffusion_r=diffusion_r,
            particle_type=particle_type,
            box_length=box_length,
            timesteps=timesteps,
            delta_t=delta_t,
            boundary_type=boundary_type,
            seed=seed,
        )

        # AVM-specific parameters
        self.tau = tau
        self.T = T
        self.D = D
        self.k = k
        self.t_a = t_a
        self.r = r

        # Desired directions (e0_i) are given by initial heading angles
        self.e0 = np.vstack(
            [np.cos(self.positions[0][2]), np.sin(self.positions[0][2])]
        ).T

    def _unit_vectors(self, theta):
        """Convert heading angle θ → unit vector e = (cosθ, sinθ)."""
        return np.vstack([np.cos(theta), np.sin(theta)]).T

    def _apply_periodic_wrapping(self, dx, dy):
        """Apply periodic boundary conditions to dx, dy."""
        if self.boundary_type[0] == BoundaryType.PERIODIC:
            dx -= self.box_length * np.round(dx / self.box_length)
        if self.boundary_type[1] == BoundaryType.PERIODIC:
            dy -= self.box_length * np.round(dy / self.box_length)
        return dx, dy

    def particle_system(self, t, positions):
        """
        Implements the full AVM:
          - Anticipation-based direction control (Eqs. 1–7)
          - Collision-free speed control (Eqs. 8–10)
        """
        positions = positions.reshape(self.n, 3).T
        x = positions[0]
        y = positions[1]
        theta = positions[2]

        # Current direction vectors e_i
        e = self._unit_vectors(theta)

        # --- Anticipation phase: predicted positions ---
        x_pred = x + self.v0 * e[:, 0] * self.t_a
        y_pred = y + self.v0 * e[:, 1] * self.t_a

        # Apply periodic boundary adjustments
        x_pred %= self.box_length
        y_pred %= self.box_length

        # Pairwise predicted displacements
        dx = x_pred[:, None] - x_pred[None, :]
        dy = y_pred[:, None] - y_pred[None, :]
        dx, dy = self._apply_periodic_wrapping(dx, dy)

        dist = np.sqrt(dx**2 + dy**2) + 1e-9
        eij = np.stack([dx / dist, dy / dist], axis=-1)

        # --- Directional dependency α_ij ---
        dot_e0_ej = np.clip(np.dot(self.e0, e.T), -1, 1)
        alpha = self.k * (1 + (1 - dot_e0_ej) / 2)

        # --- Predicted distance s^a_ij (Eq. 2) ---
        sa_ij = np.maximum(2 * self.r, (dx * eij[..., 0] + dy * eij[..., 1]))

        # --- Interaction strength R_ij (Eq. 3) ---
        R_ij = alpha * np.exp((2 * self.r - sa_ij) / self.D)

        # --- Avoidance direction n_ij (Eq. 5) ---
        e0_perp = np.stack([-self.e0[:, 1], self.e0[:, 0]], axis=1)
        sign_term = np.sign(
            np.einsum(
                "ij,ij->i",
                eij.reshape(self.n * self.n, 2),
                np.repeat(e0_perp, self.n, axis=0),
            )
        ).reshape(self.n, self.n)
        n_ij = -sign_term[..., None] * e0_perp[:, None, :]

        # --- Combined directional response (Eq. 6) ---
        R_sum = np.sum(R_ij[..., None] * n_ij, axis=1)
        e_d = self.e0 + R_sum
        e_d /= np.linalg.norm(e_d, axis=1)[:, None]

        # --- Direction evolution ODE (Eq. 7) ---
        de_dt = (e_d - e) / self.tau

        # --- Collision-free speed control (Eqs. 8–10) ---
        # Compute forward neighbors (same as Eq. 8)
        dx_curr = x[:, None] - x[None, :]
        dy_curr = y[:, None] - y[None, :]
        dx_curr, dy_curr = self._apply_periodic_wrapping(dx_curr, dy_curr)
        dist_curr = np.sqrt(dx_curr**2 + dy_curr**2) + 1e-9
        eij_curr = np.stack([dx_curr / dist_curr, dy_curr / dist_curr], axis=-1)

        forward_mask = (
            np.einsum(
                "ij,ij->i",
                np.repeat(e, self.n, axis=0),
                eij_curr.reshape(self.n * self.n, 2),
            ).reshape(self.n, self.n)
            >= 0
        )
        lateral_mask = np.abs(np.dot(e[:, None, :], eij_curr.transpose(0, 2, 1))) <= 1
        J_mask = forward_mask & lateral_mask & (dist_curr > 0)

        # Minimum forward distance to avoid overlap (Eq. 9)
        s_i = np.zeros(self.n)
        for i in range(self.n):
            valid = np.where(J_mask[i])[0]
            if len(valid) > 0:
                s_i[i] = np.min(dist_curr[i, valid] - 2 * self.r)
            else:
                s_i[i] = np.inf

        # Final speed (Eq. 10)
        v = np.minimum(self.v0, np.maximum(0.0, s_i / self.T))

        # --- Kinematic ODEs ---
        dxdt = v * e[:, 0]
        dydt = v * e[:, 1]
        dthetadt = np.cross(e, de_dt)  # scalar angular velocity (z-component)

        derivative = np.vstack([dxdt, dydt, dthetadt])
        return derivative.T.reshape(3 * self.n)


class WCA2(BaseSimulation):
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
        seed=0,
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
            seed,
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
