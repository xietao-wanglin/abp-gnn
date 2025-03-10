from models import GNN, GAT
from utilities import process_simulation_data, ParticleDataset, collate_fn

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb

import numpy as np
from tqdm import tqdm

import os
import json
from glob import glob
from typing import Optional, List, Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float
wandb.login()

def train(model: nn.Module, 
          train_simulation_list: List, 
          test_simulation_list: List,
          cluster_method: str,
          cluster_parameter: float | int,
          batch_size: Optional[int] = 1,
          n_epochs: Optional[int] = 100, 
          lr: Optional[float] = 5e-4, 
          weight_decay: Optional[float] = 1e-4,
          device: Optional[str | torch.device] = 'cpu',
          checkpoint_dir: Optional[str] = 'checkpoints',
          checkpoint_every: Optional[int] = 2,
          hist_filename: Optional[str] = 'training_history',
          subset: Optional[bool] = False,
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
        Name of training history JSON file, defualt is 'training history'.
    subset: bool, optional
        If True, use a subset of the trajectories instead of full simulations.
    
    Returns
    -------
    history: dict
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

    data_pairs_train = process_simulation_data(train_simulation_list, 
                                               cluster_method=cluster_method,
                                               p=cluster_parameter,
                                               subset=subset, 
                                               dtype=dtype, 
                                               device=device)
    data_pairs_test = process_simulation_data(test_simulation_list, 
                                              cluster_method=cluster_method,
                                              p=cluster_parameter,
                                              subset=subset, 
                                              dtype=dtype, 
                                              device=device)
    
    train_dataset = ParticleDataset(data_pairs_train)
    test_dataset = ParticleDataset(data_pairs_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss(reduction='none')
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
        'val_loss': [],
        'best_val_loss': float('inf')
    }

    initial_epoch = 0 
    if base_model_path is not None:
        params = torch.load(base_model_path, map_location=device)
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        scheduler.load_state_dict(params['scheduler_state_dict'])
        initial_epoch = params['epoch']
    
    for epoch in range(initial_epoch, n_epochs+initial_epoch):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch_idx, batch in enumerate(pbar):
            batch_loss = 0
            for x, t, res, edge_index, edge_attr in batch:
                x = x.transpose(0, 1).to(device)  # Shape: (3, N) -> (N, 3)
                res = res.transpose(0, 1).to(device) # Range: [0, 1] x [0, 1] x [0, 2pi]
                
                predictions = model(x, edge_index, edge_attr)
                loss = criterion(predictions.squeeze(), res[:, 2]).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

                wandb.log(
                    {'train_batch_loss': loss.item(), 
                     'epoch': epoch, 
                     'timestep': t, 
                     'batch': batch_idx}
                    )
            
            train_losses.append(batch_loss/len(batch))
            
            pbar.set_postfix({'train_loss': f'{batch_loss:.6f}'})
        
        scheduler.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0
                for x, _, res, edge_index, edge_attr in batch:
                    x = x.transpose(0, 1).to(device)
                    res = res.transpose(0, 1).to(device)

                    predictions = model(x, edge_index, edge_attr)
                    loss = criterion(predictions.squeeze(), res[:, 2]).mean()
                    
                    batch_loss += loss.item()
                
                val_losses.append(batch_loss/len(batch))
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
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
        
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
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

    train_glob = glob('./data_euler/simulation_train_*')[:1000]
    train_simulations = [np.load(sim) for sim in train_glob]

    test_glob = glob('./data_euler/simulation_test_*')[:200]
    test_simulations = [np.load(sim) for sim in test_glob]

    model = GNN(
        n_layers=3,
        in_node_nf=3,
        in_edge_nf=1,
        hidden_nf=64,
        dropout=0,
        device=device,
        norm=False
    ).to(dtype=dtype)

    history = train(
        model=model,
        cluster_method='radius',
        cluster_parameter=0.1,
        batch_size=1,
        train_simulation_list=train_simulations,
        test_simulation_list=test_simulations,
        n_epochs=50,
        lr=5e-4,
        weight_decay=1e-4,
        subset=True,
        device=device
    )
