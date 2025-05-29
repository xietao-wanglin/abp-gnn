import numpy as np
from tqdm import tqdm
import torch

import time

from src.continuous.models import GNN
from src.continuous.utils import compute_graph, apply_periodic_boundary

device = 'cpu'
dtype = torch.float

if __name__ == '__main__':

    model = GNN(
        n_layers=3,
        in_node_nf=3,
        in_edge_nf=1,
        hidden_nf=64,
        dropout=0,
        device=device,
        norm=False
    ).to(dtype=dtype)

    data = torch.load('./checkpoints_working/best_model.pt', map_location=device)
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    num_of_particles = np.array([2**x for x in range(1, 13)])
    n_replic = 5
    rot_rate = 1
    times = np.zeros((num_of_particles.shape[0], n_replic))
    filename = 'scaling_model_wg.dat'
    with open(filename, 'w') as f:
        f.close()

    for j, n in enumerate(tqdm(num_of_particles, desc='Power')):
        for replic in tqdm(range(n_replic), desc='Replic', leave=False):
            sim_loc = f'./data_scaling/power_{replic}_{n}.npy'
            data = torch.tensor(np.load(sim_loc), dtype=torch.float)
            init = data[0]
            predictions = torch.zeros(size=(100, 3, init.shape[1]), dtype=torch.float)
            predictions[0] = init
            total_compute_graph_time = 0
            start = time.time()
            for i in range(99):
                x = predictions[i].clone()
                cg_start = time.time()
                edge_index, edge_attr = compute_graph((x), method='radius', p=0.1)
                cg_end = time.time()
                total_compute_graph_time += (cg_end - cg_start)
                with torch.no_grad():
                    x = x.transpose(0, 1)
                    pred_theta = model(x, edge_index, edge_attr).squeeze()
                    theta = x[:, 2].squeeze()
                    pred_x = 0.1*torch.cos(theta)
                    pred_y = 0.1*torch.sin(theta)
                    pred_res = torch.cat([pred_x.unsqueeze(1), pred_y.unsqueeze(1), pred_theta.unsqueeze(1)], dim=1)
                pred = x + 0.1*pred_res
                full_pred = apply_periodic_boundary(pred.T)
                predictions[i+1] = full_pred
            end = time.time()
            net_time = (end - start)
            times[j, replic] = net_time
        with open(filename, 'ab') as f:
            np.savetxt(f, np.atleast_2d(times[j]), delimiter=',', newline = '\n')