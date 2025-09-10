import numpy as np
import torch
from glob import glob
from torch_geometric.data import Data, Batch
from src.utils import apply_periodic_boundary, compute_graph
import os
import json
from typing import Optional, Tuple

class Validator:

    def __init__(
        self,
        cluster_method: Optional[str] = "radius",
        cluster_parameter: Optional[int] = 0.1,
        use_distance: Optional[bool] = False,
        use_rel_pos: Optional[bool] = False,
        boundary_type: Optional[Tuple[int, int, int]] = (1, 1, 1),
        device: Optional[str] = "cpu",
        dtype: Optional[torch.dtype] = torch.double,
    ):
        self.cluster_method = cluster_method
        self.cluster_parameter = cluster_parameter
        self.use_distance = use_distance
        self.use_rel_pos = use_rel_pos
        self.boundary_type = boundary_type
        self.device = device
        self.dtype = dtype
    def load_data(self, dataset_name, timesteps):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sims_loc = f"{script_dir}/../datasets/{dataset_name}/data/simulation_test_*"
        with open(f"{script_dir}/../datasets/{dataset_name}/metadata.json") as f:
            self.metadata = json.load(f)
            
        sims_glob = glob(sims_loc)
        sims_list = [torch.tensor(np.load(f), device=self.device, dtype=self.dtype) for f in sims_glob]
        
        initial_states = []
        ground_truths = []
        
        for sim in sims_list:
            for t in timesteps:
                x_init = sim[t]
                x_bounded = apply_periodic_boundary(x_init)
                initial_states.append(x_bounded)
                
                gt_trajectory = apply_periodic_boundary(sim[t + 1 : t + 21])
                ground_truths.append(gt_trajectory)
        
        self.initial_states = initial_states
        self.ground_truths = ground_truths
        self.rollout_length = len(ground_truths[0])
        
        return len(initial_states)
    
    def compute_rollout(self, model):
        if self.initial_states is None:
            raise ValueError("Must call load_data first")
            
        model.eval()
        num_trajectories = len(self.initial_states)
        
        predictions = []
        for i in range(num_trajectories):
            N_i = self.initial_states[i].shape[1]
            pred_trajectory = torch.zeros(self.rollout_length + 1, 3, N_i, dtype=self.dtype)
            pred_trajectory[0] = self.initial_states[i]
            predictions.append(pred_trajectory)
        
        for t in range(self.rollout_length):
            batch_data_list = []
            trajectory_sizes = []
            
            for traj_idx in range(num_trajectories):
                x = predictions[traj_idx][t].clone()
                N_i = x.shape[1]
                trajectory_sizes.append(N_i)
                
                x_bounded = apply_periodic_boundary(x)
                
                edge_index, edge_attr = compute_graph(
                    x_bounded,
                    method=self.cluster_method,
                    p=self.cluster_parameter,
                    use_distance=self.use_distance,
                    use_rel_pos=self.use_rel_pos,
                    boundary_type=self.boundary_type,
                    device=self.device,
                )
                
                data = Data(x=x_bounded[:2].T, edge_index=edge_index, edge_attr=edge_attr)
                batch_data_list.append(data)
            
            batched_data = Batch.from_data_list(batch_data_list)
            
            with torch.no_grad():
                batched_pred = model(batched_data) * self.metadata["vel_std"] + self.metadata["vel_mean"]
            
            start_idx = 0
            for traj_idx in range(num_trajectories):
                N_i = trajectory_sizes[traj_idx]
                end_idx = start_idx + N_i
                
                vel_pred = batched_pred[start_idx:end_idx]
                
                theta_vel = torch.ones(N_i, 1) * self.metadata["angular_mean"]
                full_vel_pred = torch.cat([vel_pred, theta_vel], dim=-1)
                
                current_state = predictions[traj_idx][t].clone()
                next_state = current_state + full_vel_pred.T
                
                predictions[traj_idx][t + 1] = apply_periodic_boundary(
                    next_state
                )
                
                start_idx = end_idx
        
        self.predictions = predictions
    
    def compute_metrics(self):
        if self.predictions is None:
            raise ValueError("Must call compute_rollout first")
            
        mse_all = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_rollout = pred[1:]
            
            mse_1 = torch.mean((pred_rollout[0, :2] - gt[0, :2]) ** 2)
            mse_5 = torch.mean((pred_rollout[:5, :2] - gt[:5, :2]) ** 2)
            mse_10 = torch.mean((pred_rollout[:10, :2] - gt[:10, :2]) ** 2)
            mse_20 = torch.mean((pred_rollout[:20, :2] - gt[:20, :2]) ** 2)
            mse_all.append([mse_1.item(), mse_5.item(), mse_10.item(), mse_20.item()])
        
        mse_all = np.array(mse_all)
        metrics = {
            "mse_1": mse_all[:, 0].mean(),
            "mse_5": mse_all[:, 1].mean(),
            "mse_10": mse_all[:, 2].mean(),
            "mse_20": mse_all[:, 3].mean(),
        }
        
        return metrics
    
if __name__ == "__main__":
    val = Validator()
    length = val.load_data("nonchiral_lj", [0, 20, 40, 60])
    print(length)
    print(val.initial_states[0].shape)
    print(val.ground_truths[0].shape)