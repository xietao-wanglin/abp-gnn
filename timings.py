from src.rollout import model_rollout
from omegaconf import OmegaConf
import torch
import numpy as np
import argparse
import time

def gns_lattice95():
    experiment = "chiral"
    cfg = OmegaConf.load(f"../experiments/{experiment}/cfg.yaml")
    device = "cpu"
    dtype = torch.float
    model_step = 1_000_000
    timesteps = 16000
    gt = np.load("./table/data/lattice95.npz")
    particle_type = np.zeros(401, dtype=int)
    particle_type[-1] = 1
    particles = torch.tensor(particle_type, dtype=int)
    box_length = gt["box_length"].item()
    out = []
    total_seconds = 0
    for particle_idx in range(200):
        x = torch.tensor(gt["initial_state"], dtype=dtype)
        obstacles = x[:, :400]
        active_particle = x[:, 400 + particle_idx : 400 + particle_idx + 1]
        initial_state = torch.cat([obstacles, active_particle], dim=1)
        start = time.time()
        res = model_rollout(cfg, 
            experiment_name=experiment,
            model_step=model_step,
            timesteps=timesteps+1, 
            initial_state=initial_state, 
            box_length=box_length,
            particles=particles,
            particle_features=None,
            device=device,
            record_every=10,
            dtype=dtype)
        elapsed = time.time() - start
        total_seconds += elapsed
        out.append(res[:, :2, -1])
    data = np.stack(out, axis=-1)
    np.save("./table/data/gns_lattice95.npy", data)
    return total_seconds

def gns_lattice18():
    experiment = "chiral"
    cfg = OmegaConf.load(f"../experiments/{experiment}/cfg.yaml")
    device = "cpu"
    dtype = torch.float
    model_step = 1_000_000
    timesteps = 16000
    gt = np.load("./table/data/lattice18.npz")
    particle_type = np.zeros(401, dtype=int)
    particle_type[-1] = 1
    particles = torch.tensor(particle_type, dtype=int)
    box_length = gt["box_length"].item()
    out = []
    total_seconds = 0
    for particle_idx in range(200):
        x = torch.tensor(gt["initial_state"], dtype=dtype)
        obstacles = x[:, :400]
        active_particle = x[:, 400 + particle_idx : 400 + particle_idx + 1]
        initial_state = torch.cat([obstacles, active_particle], dim=1)
        start = time.time()
        res = model_rollout(cfg, 
            experiment_name=experiment,
            model_step=model_step,
            timesteps=timesteps+1, 
            initial_state=initial_state, 
            box_length=box_length,
            particles=particles,
            particle_features=None,
            device=device,
            record_every=10,
            dtype=dtype)
        elapsed = time.time() - start
        total_seconds += elapsed
        out.append(res[:, :2, -1])
    data = np.stack(out, axis=-1)
    np.save("./table/data/gns_lattice18.npy", data)
    return total_seconds

def egnn_lattice95():
    experiment = "chiral"
    cfg = OmegaConf.load(f"../experiments/{experiment}/cfg.yaml")
    device = "cpu"
    dtype = torch.float
    model_step = 66_000
    timesteps = 16000
    gt = np.load("./table/data/lattice95.npz")
    particle_type = np.zeros(401, dtype=int)
    particle_type[-1] = 1
    particles = torch.tensor(particle_type, dtype=int)
    box_length = gt["box_length"].item()
    out = []
    total_seconds = 0
    for particle_idx in range(200):
        x = torch.tensor(gt["initial_state"], dtype=dtype)
        obstacles = x[:, :400]
        active_particle = x[:, 400 + particle_idx : 400 + particle_idx + 1]
        initial_state = torch.cat([obstacles, active_particle], dim=1)
        start = time.time()
        res = model_rollout(cfg, 
            experiment_name=experiment,
            model_step=model_step,
            timesteps=timesteps+1, 
            initial_state=initial_state, 
            box_length=box_length,
            particles=particles,
            particle_features=None,
            device=device,
            record_every=10,
            dtype=dtype)
        elapsed = time.time() - start
        total_seconds += elapsed
        out.append(res[:, :2, -1])
    data = np.stack(out, axis=-1)
    np.save("./table/data/egnn_lattice95.npy", data)
    return total_seconds

def egnn_lattice18():
    experiment = "chiral"
    cfg = OmegaConf.load(f"../experiments/{experiment}/cfg.yaml")
    device = "cpu"
    dtype = torch.float
    model_step = 66_000
    timesteps = 16000
    gt = np.load("./table/data/lattice18.npz")
    particle_type = np.zeros(401, dtype=int)
    particle_type[-1] = 1
    particles = torch.tensor(particle_type, dtype=int)
    box_length = gt["box_length"].item()
    out = []
    total_seconds = 0
    for particle_idx in range(200):
        x = torch.tensor(gt["initial_state"], dtype=dtype)
        obstacles = x[:, :400]
        active_particle = x[:, 400 + particle_idx : 400 + particle_idx + 1]
        initial_state = torch.cat([obstacles, active_particle], dim=1)
        start = time.time()
        res = model_rollout(cfg, 
            experiment_name=experiment,
            model_step=model_step,
            timesteps=timesteps+1, 
            initial_state=initial_state, 
            box_length=box_length,
            particles=particles,
            particle_features=None,
            device=device,
            record_every=10,
            dtype=dtype)
        elapsed = time.time() - start
        total_seconds += elapsed
        out.append(res[:, :2, -1])
    data = np.stack(out, axis=-1)
    np.save("./table/data/egnn_lattice18.npy", data)
    return total_seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="index")
    args = parser.parse_args()
    index = int(args.index)
    if index == 0:
        seconds = gns_lattice95()
        print(f"GNS Lattice 9.5, took {seconds} seconds.")
    
    if index == 1:
        seconds = gns_lattice18()
        print(f"GNS Lattice 18, took {seconds} seconds.")

    if index == 2:
        seconds = egnn_lattice95()
        print(f"EGNN Lattice 9.5, took {seconds} seconds.")
    
    if index == 3:
        seconds = egnn_lattice18()
        print(f"EGNN Lattice 18, took {seconds} seconds.")