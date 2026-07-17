from src.simulation import WCA, SparseWCA

import numpy as np


def generate_state(n, box_length=1.0):
    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T

    return initial_state


def generate_state_poly(n, box_length=1.0):
    n_boundary = np.random.randint(1, 2)
    rot_rate = np.ones(n) * 1
    rot_couple = np.zeros(n)
    sigma = np.random.uniform(0.01, 0.14, size=(n,))

    initial_state = np.random.random(3 * n)
    initial_state[2::3] *= 2 * np.pi
    initial_state[0::3] = initial_state[0::3] * box_length
    initial_state[1::3] = initial_state[1::3] * box_length

    initial_state = initial_state.reshape(n, 3).T
    particle_type = np.ones(n, dtype=int)
    particle_type[:n_boundary] = 0

    return (
        rot_rate,
        rot_couple,
        sigma,
        initial_state,
        particle_type,
    )


def generate_state_chevron(n, box_length=1.0, chevron_angle=np.pi / 4, arm_length=0.4):
    n_boundary = n // 5
    n_free = n - n_boundary

    x_free = np.random.random(n_free) * box_length
    y_free = np.random.random(n_free) * box_length
    theta_free = np.random.random(n_free) * 2 * np.pi

    center = np.array([box_length / 2, box_length / 2])
    arm_points = n_boundary // 2

    s = np.linspace(0, arm_length * box_length, arm_points)
    x_left = center[0] - s * np.cos(chevron_angle)
    y_left = center[1] - s * np.sin(chevron_angle)

    x_right = center[0] + s * np.cos(chevron_angle)
    y_right = center[1] - s * np.sin(chevron_angle)

    x_bound = np.concatenate([x_left, x_right])
    y_bound = np.concatenate([y_left, y_right])
    theta_bound = np.random.random(len(x_bound)) * 2 * np.pi

    x_all = np.concatenate([x_bound, x_free])
    y_all = np.concatenate([y_bound, y_free])
    theta_all = np.concatenate([theta_bound, theta_free])

    initial_state = np.vstack([x_all, y_all, theta_all])
    particle_type = np.ones(n, dtype=int)
    particle_type[: len(x_bound)] = 0

    return initial_state, particle_type

def generate_state_fork(n_boundary_per_branch=50, branch_length=0.4, branch_angle=np.pi / 6, sigma=0.04):
    """
    Generates a Y-shaped fork geometry for a single ABP to navigate.
    
    Parameters:
    -----------
    n_boundary_per_branch : int
        Number of fixed particles forming each wall of the fork.
    branch_length : float
        The length of the fork's branches.
    branch_angle : float
        The half-angle of the fork split (in radians).
    sigma : float
        The interaction radius/size of the particles.
    """
    # Define the apex of the fork (where the split happens)
    # We will center the setup roughly in a box of size 1.0 x 1.0
    box_length = 1.0
    apex = np.array([0.4 * box_length, 0.5 * box_length])
    
    # Generate points along the upper branch wall
    s = np.linspace(0, branch_length, n_boundary_per_branch)
    x_upper = apex[0] + s * np.cos(branch_angle)
    y_upper = apex[1] + s * np.sin(branch_angle)
    
    # Generate points along the lower branch wall
    x_lower = apex[0] + s * np.cos(-branch_angle)
    y_lower = apex[1] + s * np.sin(-branch_angle)
    
    # Optional: Add a central wedge/splitter at the apex to force a clean split
    # This prevents the particle from just getting stuck dead-center on a single pixel apex
    x_wedge = apex[0] + (s[:n_boundary_per_branch // 2] * 0.5) * np.cos(0)
    y_wedge = apex[1] + np.zeros_like(x_wedge)

    # Combine all boundary elements
    x_bound = np.concatenate([x_upper, x_lower, x_wedge])
    y_bound = np.concatenate([y_upper, y_lower, y_wedge])
    theta_bound = np.zeros(len(x_bound)) # Fixed boundaries don't care about theta
    
    # Place the single Active Brownian Particle at the entrance of the channel
    # Heading straight towards the apex (theta = 0)
    active_x = apex[0] - 0.2  # Start 0.2 units back from the fork apex
    active_y = apex[1]        # Centered vertically
    active_theta = 0.0        # Pointing dead ahead towards the fork
    
    # Concatenate boundary + active particle
    all_x = np.concatenate([x_bound, [active_x]])
    all_y = np.concatenate([y_bound, [active_y]])
    all_theta = np.concatenate([theta_bound, [active_theta]])
    
    n_total = len(all_x)
    particle_type = np.zeros(n_total, dtype=int)
    particle_type[-1] = 1 # The last particle is active
    
    initial_state = np.vstack([all_x, all_y, all_theta])
    
    return particle_type, initial_state, box_length


def generate_state_with_grid_boundary(lc, n_boundary=400, sigma=0.04):
    n_side = int(np.sqrt(n_boundary))
    box_length = n_side * lc

    x = np.linspace(0, box_length - lc, n_side) + lc / 2
    y = np.linspace(0, box_length - lc, n_side) + lc / 2
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_boundary], Y.flatten()[:n_boundary]

    while True:
        active_x = np.random.rand() * box_length
        active_y = np.random.rand() * box_length
        d = np.sqrt((X - active_x) ** 2 + (Y - active_y) ** 2)
        if np.all(d > 1 * sigma):
            break
    active_theta = np.random.rand() * 2 * np.pi

    all_x = np.concatenate([X, [active_x]])
    all_y = np.concatenate([Y, [active_y]])
    all_theta = np.concatenate([np.zeros(n_boundary), [active_theta]])

    n_total = n_boundary + 1
    particle_type = np.zeros(n_total, dtype=int)
    particle_type[-1] = 1

    initial_state = np.vstack([all_x, all_y, all_theta])
    return particle_type, initial_state, box_length


def load_from_file(filepath):
    pos = np.load(filepath)
    return pos[0]


if __name__ == "__main__":
    np.random.seed(12)
    sigma = 0.04
    
    particle_type, initial_state, box_length = generate_state_fork(
        n_boundary_per_branch=10, 
        branch_length=0.4, 
        branch_angle=np.pi / 4, # 45 degree split
        sigma=sigma
    )
    sim = WCA(
            initial_state=initial_state,
            v0=0.4,             # Active self-propulsion velocity
            diffusion_r=0.01,     # Rotational diffusion coefficient (causes branching variance)
            diffusion_t=0.0,     # Neglect translational diffusion to isolate rotational effects
            rot_rate=0.0,
            sigma=sigma,
            epsilon=0.001,         # Soft-repulsive wall hardness
            timesteps=400,
            couple_radius=0.0,
            rot_couple=0.0,
            delta_t=0.01,
            box_length=box_length,
            record_every=10,
            start_record=0,
            particle_type=particle_type,
        )
    sim.solve_dynamics(method="RK45", debug=True)
    _times, loc = sim.get_solution()
    sim.create_animation()
