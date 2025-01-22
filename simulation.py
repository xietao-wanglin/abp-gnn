import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional
from tqdm import tqdm

class Simulation:

    def __init__(self, N: Optional[int] = 8,
                 v0: Optional[float] = 1.0,
                 rot_rate: Optional[np.ndarray | int] = None,
                 rot_couple: Optional[float] = 0.1,
                 couple_radius: Optional[float] = 0.1, 
                 L_box: Optional[float] = 1.0,
                 timesteps: Optional[int] = 100,
                 t_max: Optional[float] = 10.0,
                 seed: Optional[int] = 0):
        
        self.N = N
        self.v0 = v0
        if rot_rate is None:
            self.rot_rate = np.ones(N)
        if isinstance(rot_rate, int):
            self.rot_rate = np.ones(N)*rot_rate
        else:
            self.rot_rate = rot_rate
        self.rot_couple = rot_couple
        self.couple_radius = couple_radius
        self.L_box = L_box
        
        self.timesteps = timesteps
        self.t_max= t_max

        np.random.seed(seed)
        initial_state = np.random.random(3*N)
        initial_state[2::3] = initial_state[2::3]*2*np.pi
        initial_state = initial_state.reshape(N, 3).T
        self.positions = np.zeros(shape=(self.timesteps, 3, self.N))
        self.positions[0] = initial_state
        self.times = np.linspace(0, self.t_max, self.timesteps)
        self.dt = self.times[1]-self.times[0]

    def particle_system(self, t, positions):
        positions = positions.reshape(self.N, 3).T
        x = positions[0]
        y = positions[1]
        theta = positions[2]
        dxdt = self.v0*np.cos(theta)
        dydt = self.v0*np.sin(theta)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dx = dx - self.L_box * np.round(dx/self.L_box)  # Periodic boundary for x
        dy = dy - self.L_box  * np.round(dy/self.L_box)  # Periodic boundary for y
        distances = np.sqrt(dx**2 + dy**2)

        neighbor_mask = (distances < self.couple_radius) & (distances > 0)

        interaction = np.sum(
            neighbor_mask * np.sin(theta[None, :] - theta[:, None]), axis=1
        )

        dthetadt = self.rot_rate + self.rot_couple * interaction
        derivative = np.vstack([dxdt, dydt, dthetadt])
        return derivative.T.reshape(3*self.N)

    def apply_periodic_boundary(self, positions):
        positions[::3] = positions[::3] % self.L_box
        positions[1::3] = positions[1::3] % self.L_box
        positions[2::3] = positions[2::3] % (2*np.pi)
        return positions
    
    def solve_dynamics(self, method: Optional[str] = 'RK45'):

        for i, t in enumerate(tqdm(self.times[:-1], leave=False)):
            sol = solve_ivp(self.particle_system, 
                            t_span=(t, t+self.dt), 
                            y0=self.positions[i].T.reshape(3*self.N),
                            t_eval=[t+self.dt], 
                            method=method)
            next_state = sol.y[:, -1]
            next_state = self.apply_periodic_boundary(next_state)
            self.positions[i+1] = next_state.reshape(self.N, 3).T
    
    def get_solution(self):
        return self.times, self.positions
    
    def create_animation(self, filename: Optional[str] = None):
        times, positions = self.get_solution()
        f = plt.figure(figsize=(6, 5))
        ax = f.add_subplot(111)
        ax.set_xlim(0, self.L_box)
        ax.set_ylim(0, self.L_box)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(r'Time: 0.0')
        points = ax.scatter(positions[0][0], positions[0][1], c=positions[0][2], vmin=0, vmax=2*np.pi, alpha=0.3)
        f.colorbar(points, label=r'Orientation $\theta_i$')
        def update(fn):
            ax.set_title(fr'Time: {times[fn]:2f}')
            points.set_offsets(np.c_[positions[fn][0], positions[fn][1]])
            points.set_array(positions[fn][2])
            
            f.canvas.draw_idle()

        animation = FuncAnimation(f, update, interval=50, frames=100)
        if filename is not None:
            animation.save(f'./videos/{filename}', writer='ffmpeg', fps=20)
        plt.show()
