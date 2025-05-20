from models import GNN, GAT, LatentGNN
from utils import process_simulation_data, ParticleDataset, RelativeL2Loss

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import wandb

import numpy as np
from tqdm import tqdm

import os
import json
from glob import glob
from typing import Optional, List, Dict

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device} device')
dtype = torch.float
torch.manual_seed(0)
wandb.login()

def train(model: nn.Module, 
          cluster_method: str,
          cluster_parameter: float | int, 
          data_loc: str,
          batch_size: Optional[int] = 32,
          n_epochs: Optional[int] = 100, 
          lr: Optional[float] = 5e-4, 
          weight_decay: Optional[float] = 1e-4,
          device: Optional[str | torch.device] = 'cpu',
          checkpoint_dir: Optional[str] = 'checkpoints',
          checkpoint_every: Optional[int] = 2,
          hist_filename: Optional[str] = 'training_history',
          base_model_path: Optional[str] = None) -> Dict:
    """
    Train the GNN model.
    
    Parameters
    ----------
    model: nn.Module
        Model instance.
    train_simulation_list: List
        List of train simulation arrays.
    test_simulation_list: List
        List of test simulation arrays.
    cluster_method: str
        Graph creation method, either 'radius' or 'knn'.
    cluster_parameter: float or int
        Parameter used in `cluster_method`.
    n_epochs: int, optional
        Number of epochs to train, default is 100.
    lr: float, optional
        Learning rate of AdamW optimiser, default is 5e-4.
    weight_decay: float, optional
        Weight decay of AdamW optimiser, default is 1e-4.
    device: str or torch.device, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.
    checkpoint_dir: str, optional 
        Directory to save checkpoints, default is 'checkpoints'.
    checkpoint_every: int, optional
        Save checkpoint every N epochs, default is 10.
    hist_filename: str, optional
        Name of training history JSON file, defualt is 'training_history'.
    subset: bool, optional
        If True, use a subset of the trajectories instead of full simulations, default is False.
    subset_samples: List, optional
        If provided, samples to choose from trajectories if `subset=True`, otherwise random, default is None.
    base_model_path: str, optional
        If provided, use path model to load weights, default is None.
    
    Returns
    -------
    history: Dict
        Training history.
    """
    
    wandb.init(
        project='ABP_GNN', 
        config={
            'lr': lr,
            'n_epochs': n_epochs,
            'weight_decay': weight_decay,
            'model': str(model),
        }
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_glob = sorted(glob(f'./{data_loc}/simulation_train_*'))
    test_glob = sorted(glob(f'./{data_loc}/simulation_test_*'))
    times_train_glob = sorted(glob(f'./{data_loc}/times_train_*'))
    times_test_glob = sorted(glob(f'./{data_loc}/times_test_*'))

    train_simulations = [np.load(sim) for sim in train_glob]
    test_simulations = [np.load(sim) for sim in test_glob]

    train_times = [np.load(times) for times in times_train_glob]
    test_times = [np.load(times) for times in times_test_glob]

    data_pairs_train = process_simulation_data(train_simulations,
                                               train_times, 
                                               cluster_method=cluster_method,
                                               p=cluster_parameter,
                                               dtype=dtype, 
                                               device=device)
    data_pairs_test = process_simulation_data(test_simulations,
                                              test_times, 
                                              cluster_method=cluster_method,
                                              p=cluster_parameter,
                                              dtype=dtype, 
                                              device=device)
    
    train_dataset = ParticleDataset(data_pairs_train)
    test_dataset = ParticleDataset(data_pairs_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, threshold=1e-4)
    criterion = nn.MSELoss(reduction='none')
    metric = nn.L1Loss(reduction='none')
    train_details = {
        'optimizer': str(optimizer),
        'scheduler': str(scheduler),
        'criterion': str(criterion),
        'model': str(model)
    }

    details_path = os.path.join(checkpoint_dir, f'details.json')
    with open(details_path, 'w') as f:
        json.dump(train_details, f, indent=4)
    
    print(f'Training details saved to {details_path}')
    
    history = {
        'train_loss': [],
        'train_metric': [],
        'val_loss': [],
        'val_metric': [],
        'best_val_loss': float('inf')
    }

    initial_epoch = 0 
    if base_model_path is not None:
        params = torch.load(base_model_path, map_location=device, weights_only=False)
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        scheduler.load_state_dict(params['scheduler_state_dict'])
        initial_epoch = params['epoch']
    
    for epoch in range(initial_epoch, n_epochs+initial_epoch):
        model.train()
        train_losses = []
        train_metrics = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs+initial_epoch}')
        for batch_idx, batch in enumerate(pbar):

            batch = batch.to(device)
            
            y = batch.y
            t = batch.t
            pred = model(batch, t)
            loss = criterion(pred, y).mean()
            metric_train = metric(pred, y).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log(
                {'train_batch_loss': loss.item(),
                 'train_batch_metric': metric_train.item(), 
                    'epoch': epoch, 
                    'batch': batch_idx}
                )
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            train_losses.append(loss.item())
            train_metrics.append(metric_train.item())
        
        model.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y
                t = batch.t
                pred = model(batch, t)
                loss = criterion(pred, y).mean()
                metric_val = metric(pred, y).mean()
                val_losses.append(loss.item())
                val_metrics.append(metric_val.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        avg_train_metric = np.mean(train_metrics)
        avg_val_metric = np.mean(val_metrics)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_metric'].append(avg_train_metric)
        history['val_metric'].append(avg_val_metric)
        #scheduler.step(avg_val_loss)
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_metric': avg_train_metric, 
            'val_loss': avg_val_loss,
            'val_metric': avg_val_metric,
            'lr': scheduler.get_last_lr()[0]
        })
        
        if avg_val_loss < history['best_val_loss']:
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pt')
            history['best_val_loss'] = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            wandb.save(checkpoint_path)
        
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            wandb.save(checkpoint_path)
        
        print(f'\nEpoch {epoch+1}/{n_epochs+initial_epoch}')
        print(f'Train Loss: {avg_train_loss:.8f}')
        print(f'Val Loss: {avg_val_loss:.8f}')
        print('-' * 30)
    
        history_path = os.path.join(checkpoint_dir, f'{hist_filename}.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'best_val_loss': history['best_val_loss']
            }, f, indent=4)
        
        print(f'Training history saved to {history_path}')
    
    wandb.finish()
    return history

if __name__ == '__main__':

    model = LatentGNN(
        n_layers=4,
        in_node_nf=3,
        out_node_nf=2,
        latent_nf=64,
        in_edge_nf=0,
        hidden_nf=16,
        device=device,
        norm=False,
        activation=nn.SiLU()
    ).to(dtype=dtype)

    history = train(
        model=model,
        cluster_method='radius',
        cluster_parameter=0.1,
        data_loc='data',
        batch_size=1,
        checkpoint_every=100,
        n_epochs=10000,
        lr=1e-5,
        weight_decay=1e-8,
        device=device,
        base_model_path=None
    )
