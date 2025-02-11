from models import GNN_vel, GAT
from utilities import process_simulation_data, ParticleDataset, collate_fn

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

import os
import json
from glob import glob
from typing import Optional, List, Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

def train(model: nn.Module, 
          train_simulation_list: List, 
          test_simulation_list: List,
          batch_size: Optional[int] = 1,
          n_epochs: Optional[int] = 100, 
          lr: Optional[float] = 5e-4, 
          weight_decay: Optional[float] = 1e-4,
          device: Optional[str | torch.device] = 'cpu',
          checkpoint_dir: Optional[str] = 'checkpoints',
          checkpoint_every: Optional[int] = 2,
          hist_filename: Optional[str] = 'training_history',
          subset: Optional[bool] = False,
          weights: Optional[torch.Tensor] = None) -> Dict:
    """
    Train the GNN model.
    
    Parameters
    ----------
    model: nn.Module
        GNN model instance.
    train_simulation_list: List
        List of train simulation arrays.
    test_simulation_list: List
        List of test simulation arrays.
    n_epochs: int, optional
        Number of epochs to train, default is 100.
    lr: float, optional
        Learning rate of AdamW optimiser, default is 5e-4.
    weight_decay: float, optional
        Weight decay of AdamW optimiser, default is 1e-4.
    device: str, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.
    checkpoint_dir: str, optional 
        Directory to save checkpoints, default is 'checkpoints'.
    checkpoint_every: int, optional
        Save checkpoint every N epochs, default is 10.
    
    Returns
    -------
    history: dict
        Training history.
    """

    if weights is None:
        weights = torch.tensor([1, 0])

    os.makedirs(checkpoint_dir, exist_ok=True)

    data_pairs_train = process_simulation_data(train_simulation_list, subset=subset, dtype=dtype, device=device)
    data_pairs_test = process_simulation_data(test_simulation_list, subset=subset, dtype=dtype, device=device)
    
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
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss(reduction='mean')

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
        'train_loss_res':[],
        'train_loss_total':[],
        'val_loss': [],
        'val_loss_res': [],
        'val_loss_total': [],
        'best_val_loss_total': float('inf')
    }
    
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        train_losses_res = []
        train_losses_total = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            batch_loss = 0
            batch_loss_res = 0
            batch_loss_total = 0
            # Each batch contains a single (x, y) pair due to batch_size=1
            for x, y, res, edge_index, edge_attr in batch:
                x = x.transpose(0, 1).to(device)
                y = y.transpose(0, 1).to(device) # Shape: (6, N) -> (N, 6)
                res = res.transpose(0, 1).to(device)
                
                predictions_res = model(x, edge_index, edge_attr)
                predictions = predictions_res + x

                loss_res = criterion(predictions_res, res)
                loss = criterion(predictions, y)
                total_loss = weights[1]*loss_res + weights[0]*loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_loss_res += loss_res.item()
                batch_loss += loss.item()
                batch_loss_total += total_loss.item()
            
            # Average loss over samples in batch (though batch_size=1 here)
            train_losses.append(batch_loss/len(batch))
            train_losses_res.append(batch_loss_res/len(batch))
            train_losses_total.append(batch_loss_total/len(batch))
            
            pbar.set_postfix({'train_loss': f'{batch_loss_total:.6f}'})
        
        scheduler.step()
        # Validation phase
        model.eval()
        val_losses = []
        val_losses_res = []
        val_losses_total = []
        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0
                batch_loss_res = 0
                batch_loss_total = 0
                for x, y, res, edge_index, edge_attr in batch:
                    x = x.transpose(0, 1).to(device)
                    y = y.transpose(0, 1).to(device)
                    res = res.transpose(0, 1).to(device)

                    predictions_res = model(x, edge_index, edge_attr)
                    predictions = predictions_res + x

                    loss_res = criterion(predictions_res, res)
                    batch_loss_res += loss_res.item()
                    
                    loss = criterion(predictions, y)
                    batch_loss += loss.item()

                    total_loss = weights[1]*loss_res + weights[0]*loss
                    batch_loss_total += total_loss.item()
                
                val_losses.append(batch_loss/len(batch))
                val_losses_res.append(batch_loss_res/len(batch))
                val_losses_total.append(batch_loss_total/len(batch))
        
        avg_train_loss = np.mean(train_losses)
        avg_train_loss_res = np.mean(train_losses_res)
        avg_train_loss_total = np.mean(train_losses_total)
        avg_val_loss = np.mean(val_losses)
        avg_val_loss_res = np.mean(val_losses_res)
        avg_val_loss_total = np.mean(val_losses_total)
        
        history['train_loss'].append(avg_train_loss)
        history['train_loss_res'].append(avg_train_loss_res)
        history['train_loss_total'].append(avg_train_loss_total)
        history['val_loss'].append(avg_val_loss)
        history['val_loss_res'].append(avg_val_loss_res)
        history['val_loss_total'].append(avg_val_loss_total)
        
        if avg_val_loss_total < history['best_val_loss_total']:
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pt')
            history['best_val_loss_total'] = avg_val_loss_total
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
        
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_total': avg_train_loss_total,
                'val_loss_total': avg_val_loss_total,
            }, checkpoint_path)
        
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {avg_train_loss:.8f}')
        print(f'Train Loss (Residual): {avg_train_loss_res:.8f}')
        print(f'Train Loss (Total): {avg_train_loss_total:.8f}')
        print(f'Val Loss: {avg_val_loss:.8f}')
        print(f'Val Loss (Residual): {avg_val_loss_res:.8f}')
        print(f'Val Loss (Total): {avg_val_loss_total:.8f}')
        print('-' * 30)
    
    history_path = os.path.join(checkpoint_dir, f'{hist_filename}.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'train_loss_res': history['train_loss_res'],
            'train_loss_total': history['train_loss_total'],
            'val_loss': history['val_loss'],
            'val_loss_res': history['val_loss_res'],
            'val_loss_total': history['val_loss_total'],
            'best_val_loss_total': history['best_val_loss_total']
        }, f, indent=4)
    
    print(f'Training history saved to {history_path}')
    
    return history

def evaluate_model(model: nn.Module, 
                   simulation_list: List, 
                   device: Optional[str | torch.device] = 'cpu') -> float:
    """
    Evaluate the model on test data
    
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
    data_pairs = process_simulation_data(simulation_list)
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

    train_glob = glob('./data_abs/simulation_train_*')[:1000]
    train_simulations = [np.load(sim) for sim in train_glob]

    test_glob = glob('./data_abs/simulation_test_*')[:200]
    test_simulations = [np.load(sim) for sim in test_glob]

    model = GNN_vel(
        n_layers=3,
        in_node_nf=3,
        in_edge_nf=1,
        hidden_nf=64,
        dropout=0,
        device=device,
        norm=True
    ).to(dtype=dtype)

    history = train(
        model=model,
        batch_size=1,
        train_simulation_list=train_simulations,
        test_simulation_list=test_simulations,
        n_epochs=50,
        lr=5e-4,
        weight_decay=1e-4,
        subset=True,
        device=device,
        weights=torch.tensor([0, 1])
    )

    test_mse = evaluate_model(model, test_simulations, device=device)
    print(f'Test MSE: {test_mse:.6f}')
