from models import GNN, GAT
from utilities import process_simulation_data, ParticleDataset, collate_fn

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb
from torch_scatter import scatter_add

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
          weights: Optional[torch.Tensor] = None,
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

    if weights is None:
        weights = torch.tensor([1, 1, 1], device=device)
    
    wandb.init(
        project='ABP_GNN', 
        config={
            'lr': lr,
            'n_epochs': n_epochs,
            'weight_decay': weight_decay,
            'model': str(model),
            'loss_weights': weights
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
        'train_loss_x': [],
        'train_loss_y': [],
        'train_loss_theta': [],
        'val_loss': [],
        'val_loss_x': [],
        'val_loss_y': [],
        'val_loss_theta': [],
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
        train_losses_x = []
        train_losses_y = []
        train_losses_theta = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch_idx, batch in enumerate(pbar):
            batch_loss = 0
            batch_loss_x = 0
            batch_loss_y = 0
            batch_loss_theta = 0
            for x, t, res, edge_index, edge_attr in batch:
                x = x.transpose(0, 1).to(device)  # Shape: (3, N) -> (N, 3)
                res = res.transpose(0, 1).to(device) # Range: [0, 1] x [0, 1] x [0, 2pi]
                
                predictions = model(x, edge_index, edge_attr)
    
                # Initialize output tensor with ones since T(i) starts with 1
                source, target = edge_index  # source = j, target = i

                # Aggregate edge features h_{ij} for each target node i
                agg_h = scatter_add(edge_attr, target, dim=0, dim_size=x.shape[0])

                # Compute T(i)
                T = 1 + 0.1 * agg_h
                T[agg_h == 0] = 1
                
                loss_x = (weights[0]*criterion(predictions.squeeze(), res[:, 0])).mean()
                loss_x = criterion(predictions, T).mean()
                loss_y = (weights[1]*criterion(predictions.squeeze(), res[:, 1])).mean()
                loss_y = criterion(res[:, 2].squeeze(), T.squeeze()).mean()
                loss_theta = (weights[2]*criterion(predictions.squeeze(), res[:, 2])).mean()
                loss = (loss_x + 0*loss_y + 0*loss_theta)/3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                batch_loss_x += loss_x.item()
                batch_loss_y += loss_y.item()
                batch_loss_theta += loss_theta.item()

                wandb.log(
                    {'train_batch_loss': loss.item(), 
                     'train_batch_loss_x': loss_x.item(), 
                     'train_batch_loss_y': loss_y.item(), 
                     'train_batch_loss_theta': loss_theta.item(), 
                     'epoch': epoch, 
                     'timestep': t, 
                     'batch': batch_idx}
                    )
            
            # Average loss over samples in batch (though batch_size=1 here)
            train_losses.append(batch_loss/len(batch))
            train_losses_x.append(batch_loss_x/len(batch))
            train_losses_y.append(batch_loss_y/len(batch))
            train_losses_theta.append(batch_loss_theta/len(batch))
            
            pbar.set_postfix({'train_loss': f'{batch_loss:.6f}'})
        
        scheduler.step()
        # Validation phase
        model.eval()
        val_losses = []
        val_losses_x = []
        val_losses_y = []
        val_losses_theta = []
        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0
                batch_loss_x = 0
                batch_loss_y = 0
                batch_loss_theta = 0
                for x, y, res, edge_index, edge_attr in batch:
                    x = x.transpose(0, 1).to(device)
                    res = res.transpose(0, 1).to(device)

                    predictions = model(x, edge_index, edge_attr)

                    loss_x = (weights[0]*criterion(predictions.squeeze(), res[:, 0])).mean()
                    loss_y = (weights[1]*criterion(predictions.squeeze(), res[:, 1])).mean()
                    loss_theta = (weights[2]*criterion(predictions.squeeze(), res[:, 2])).mean()
                    loss = (loss_x + loss_y + loss_theta)/3
                    
                    batch_loss += loss.item()
                    batch_loss_x += loss_x.item()
                    batch_loss_y += loss_y.item()
                    batch_loss_theta += loss_theta.item()
                
                val_losses.append(batch_loss/len(batch))
                val_losses_x.append(batch_loss_x/len(batch))
                val_losses_y.append(batch_loss_y/len(batch))
                val_losses_theta.append(batch_loss_theta/len(batch))
        
        avg_train_loss = np.mean(train_losses)
        avg_train_loss_x = np.mean(train_losses_x)
        avg_train_loss_y = np.mean(train_losses_y)
        avg_train_loss_theta = np.mean(train_losses_theta)
        avg_val_loss = np.mean(val_losses)
        avg_val_loss_x = np.mean(val_losses_x)
        avg_val_loss_y = np.mean(val_losses_y)
        avg_val_loss_theta = np.mean(val_losses_theta)
        
        history['train_loss'].append(avg_train_loss)
        history['train_loss_x'].append(avg_train_loss_x)
        history['train_loss_y'].append(avg_train_loss_y)
        history['train_loss_theta'].append(avg_train_loss_theta)
        history['val_loss'].append(avg_val_loss)
        history['val_loss_x'].append(avg_val_loss_x)
        history['val_loss_y'].append(avg_val_loss_y)
        history['val_loss_theta'].append(avg_val_loss_theta)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_loss_x': avg_train_loss_x,
            'train_loss_y': avg_train_loss_y,
            'train_loss_theta': avg_train_loss_theta,
            'val_loss': avg_val_loss,
            'val_loss_x': avg_val_loss_x,
            'val_loss_y': avg_val_loss_y,
            'val_loss_theta': avg_val_loss_theta,
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
        print(f'Train Loss (x): {avg_train_loss_x:.6f}')
        print(f'Train Loss (y): {avg_train_loss_y:.6f}')
        print(f'Train Loss (theta): {avg_train_loss_theta:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print(f'Val Loss (x): {avg_val_loss_x:.6f}')
        print(f'Val Loss (y): {avg_val_loss_y:.6f}')
        print(f'Val Loss (theta): {avg_val_loss_theta:.6f}')
        print('-' * 30)
    
        history_path = os.path.join(checkpoint_dir, f'{hist_filename}.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_loss': history['train_loss'],
                'train_loss_x': history['train_loss_x'],
                'train_loss_y': history['train_loss_y'],
                'train_loss_theta': history['train_loss_theta'],
                'val_loss': history['val_loss'],
                'val_loss_x': history['val_loss_x'],
                'val_loss_y': history['val_loss_y'],
                'val_loss_theta': history['val_loss_theta'],
                'best_val_loss': history['best_val_loss']
            }, f, indent=4)
        
        print(f'Training history saved to {history_path}')
    
    wandb.finish()
    return history

def evaluate_model(model: nn.Module, 
                   simulation_list: List, 
                   cluster_method: str,
                   cluster_parameter: float | int,
                   device: Optional[str | torch.device] = 'cpu') -> float:
    """
    TODO: Evaluate the model on test data
    
    Parameters
    ----------
    model: nn.Module
        Trained GNN model.
    simulation_list: List
        List of test simulation arrays.
    device: str, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.
    
    Returns
    -------
    mse: float
        Mean squared error on test data
    """
    model.eval()
    data_pairs = process_simulation_data(simulation_list, 
                                         cluster_method=cluster_method,
                                         p=cluster_parameter)
    dataset = ParticleDataset(data_pairs)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    total_loss = 0
    criterion = nn.MSELoss()
    n_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            for x, y, res, edge_index, edge_attr in batch:
                x = x.transpose(0, 1).to(device)
                y = y.transpose(0, 1).to(device)

                predictions = model(x, edge_index, edge_attr)

                loss = criterion(predictions, y)
                
                total_loss += loss.item()
                n_samples += 1
    
    return total_loss / n_samples

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

#    model = GAT(
#        n_layers=3,
#        in_node_nf=3,
#        in_edge_nf=1,
#        hidden_nf=128,
#        dropout=0,
#        device=device,
#        norm=True
#    ).to(dtype=dtype)

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
        device=device,
        weights=torch.tensor([0, 0, 1])
    )

    #test_mse = evaluate_model(model, test_simulations, device=device, cluster_method='radius', cluster_parameter=0.6)
    #print(f'Test MSE: {test_mse:.6f}')
