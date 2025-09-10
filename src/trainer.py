from src.models import GNN
from src.utils import (
    discrete_simulation,
    ParticleDataset,
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
import wandb

import numpy as np
from tqdm import tqdm

import os
import json
import yaml
import random
from glob import glob

dtype = torch.double

class Trainer:

    def __init__(self, config):
        self.cfg = yaml.load(open(config))
        if self.cfg.seed is None:
            self.seed = random.randint(0, 2**31)
        self.seed = self.cfg.seed
        self.set_seed(self.seed)

        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_dir = os.path.join(self.script_dir, "ckp")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.load_data(self.cfg.dataset)
        self.model = self.create_model()

        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = None
        self.criterion = nn.MSELoss(reduction="none")
        self.metric = nn.L1Loss(reduction="none")
    
    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_data(self, dataset):
        
        train_glob = sorted(
            glob(f"{self.script_dir}/../../datasets/{dataset}/data/simulation_train_*")
        )
        test_glob = sorted(
            glob(f"{self.script_dir}/../../datasets/{dataset}/data/simulation_test_*")
        )
        with open(f"{self.script_dir}/../../datasets/{dataset}/metadata.json") as f:
            self.metadata = json.load(f)
        
        self.train_simulations = [torch.tensor(np.load(f), device=self.device, dtype=self.dtype) for f in train_glob]
        self.test_simulations = [torch.tensor(np.load(f), device=self.device, dtype=self.dtype) for f in test_glob]

        data_pairs_train = discrete_simulation(
            self.train_simulations,
            subset=self.cfg.subset,
            subset_samples=self.cfg.subset_samples,
            cluster_method=self.cfg.cluster_method,
            p=self.cfg.cluster_parameter,
            use_distance=self.cfg.use_distance,
            use_rel_pos=self.cfg.use_rel_pos,
            use_pos=self.cfg.use_pos,
            target_vel=self.cfg.target_vel,
            stats=self.metadata,
            boundary_type=self.cfg.boundary_type,
            dtype=dtype,
            device=self.device,
        )
        data_pairs_test = discrete_simulation(
            self.test_simulations,
            subset=self.cfg.subset,
            subset_samples=self.cfg.subset_samples,
            cluster_method=self.cfg.cluster_method,
            p=self.cfg.cluster_parameter,
            use_distance=self.cfg.use_distance,
            use_rel_pos=self.cfg.use_rel_pos,
            use_pos=self.cfg.use_pos,
            target_vel=self.cfg.target_vel,
            boundary_type=self.cfg.boundary_type,
            stats=self.metadata,
            dtype=dtype,
            device=self.device,
        )

        train_dataset = ParticleDataset(data_pairs_train)
        test_dataset = ParticleDataset(data_pairs_test)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
        )

    def create_model(self):
        model_type = self.cfg.model.name
        if model_type == "GNN":
            model = GNN(
                n_layers=self.cfg.model.n_layers,
                in_node_nf=self.cfg.model.in_node_nf,
                out_node_nf=self.cfg.model.out_node_nf,
                in_edge_nf=self.cfg.model.in_edge_nf,
                hidden_nf=self.cfg.model.hidden_nf,
                device=self.device,
                dropout=self.cfg.model.dropout,
                norm=self.cfg.model.norm
            )
        else:
            print("Model name not valid.")
        return model
