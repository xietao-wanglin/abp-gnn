import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional

class Simulation:

    def __init__(self, N: Optional[int] = 8,
                 v0: Optional[float] = 1.0,
                 rot_rate: Optional[float] = 1.0,
                 L_box: Optional[float] = 1.0,
                 seed: Optional[int] = 0):
        
        self.N = N
        self.v0 = v0
        self.rot_rate = rot_rate
        self.L_box = L_box

        np.random.seed(seed)
        initial_state = np.random.random(3*N)
        initial_state[2::3] = initial_state[2::3]*2*np.pi
        initial_state = initial_state.reshape(N, 3).T
        self.positions = [initial_state]
        self.times = [0]

    def particle_system(self, t, positions):
        N = positions.shape[0] // 3
        positions = positions.reshape(N, 3).T
        theta = positions[2]
        dxdt = self.v0*np.cos(theta)
        dydt = self.v0*np.sin(theta)
        dthetadt = self.rot_rate*np.ones(N)
        derivative = np.vstack([dxdt, dydt, dthetadt])
        return derivative.T.reshape(3*N)

    def apply_periodic_boundary(self, positions):
        positions[::3] = positions[::3] % self.L_box
        positions[1::3] = positions[1::3] % self.L_box
        positions[2::3] = positions[2::3] % (2*np.pi)
        return positions
    
    def solve_dynamics(self, timesteps: Optional[int] = 100,
                       t_max: Optional[float] = 1.0,
                       method: Optional[str] = 'RK45'):
        t_eval = np.linspace(0, t_max, timesteps)
        dt = t_eval[1]-t_eval[0]
        for t in t_eval[:-1]:
            sol = solve_ivp(self.particle_system, (t, t+dt), 
                            self.positions[-1].T.reshape(3*self.N),
                            t_eval=[t + dt], 
                            method=method)
            next_state = sol.y[:, -1]
            next_state = self.apply_periodic_boundary(next_state)
            self.positions.append(next_state.reshape(self.N, 3).T)
            self.times.append(t + dt)
    
    def get_solution(self):
        return np.array(self.times), np.array(self.positions)
    
    def create_animation(self):
        times, positions = self.get_solution()
        f = plt.figure(figsize=(6, 5))
        ax = f.add_subplot(111)
        ax.set_xlim(0, self.L_box)
        ax.set_ylim(0, self.L_box)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(r'Time: 0.0')
        points = ax.scatter(positions[0][0], positions[0][1], c=positions[0][2], vmin=0, vmax=2*np.pi)
        f.colorbar(points, label=r'$\theta_i$')
        def update(fn):
            ax.set_title(fr'Time: {times[fn]:2f}')
            points.set_offsets(np.c_[positions[fn][0], positions[fn][1]])
            points.set_array(positions[fn][2])
            
            f.canvas.draw_idle()

        animation = FuncAnimation(f, update, interval=50, frames=100)
        plt.show()
