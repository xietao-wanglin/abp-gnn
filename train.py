from models import GNN, GAT
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
          subset: Optional[bool] = False) -> Dict:
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
    criterion = nn.MSELoss(reduction='none')
    weights = torch.tensor([1, 1, 1, 1, 4*np.pi*np.pi, 4*np.pi*np.pi], device=device)

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
    
    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            batch_loss = 0
            # Each batch contains a single (x, y) pair due to batch_size=1
            for x, y, res, edge_index, edge_attr in batch:
                x = x.transpose(0, 1).to(device)
                y = y.transpose(0, 1).to(device) # Shape: (6, N) -> (N, 6)
                
                predictions = model(x, edge_index, edge_attr)

                loss = criterion(predictions, y)
                loss = (weights*loss).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
            
            train_losses.append(batch_loss/len(batch))
            
            pbar.set_postfix({'train_loss': f'{batch_loss:.6f}'})
        
        scheduler.step()
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0
                for x, y, res, edge_index, edge_attr in batch:
                    x = x.transpose(0, 1).to(device)
                    y = y.transpose(0, 1).to(device)

                    predictions = model(x, edge_index, edge_attr)
                    
                    loss = criterion(predictions, y)
                    loss = (weights*loss).mean()
                    batch_loss += loss.item()
                
                val_losses.append(batch_loss/len(batch))
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if avg_val_loss < history['best_val_loss']:
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pt')
            history['best_val_loss'] = avg_val_loss
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
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        
        print(f'\nEpoch {epoch+1}/{n_epochs}')
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

    train_glob = glob('./data_medium/simulation_train_*')[:1000]
    train_simulations = [np.load(sim) for sim in train_glob]

    test_glob = glob('./data_medium/simulation_test_*')[:200]
    test_simulations = [np.load(sim) for sim in test_glob]

    model = GNN(
        n_layers=3,
        in_node_nf=6,
        in_edge_nf=1,
        hidden_nf=64,
        dropout=0,
        device=device,
        force_circle=True
    ).to(dtype=dtype)

    #    model = GAT(
    #        n_layers=3,
    #        in_node_nf=6,
    #        edge_dim=1,
    #        hidden_nf=64,
    #        dropout=0,
    #        device=device,
    #        force_circle=True,
    #        heads=4,
    #    ).to(dtype=dtype)

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
    )

    test_mse = evaluate_model(model, test_simulations, device=device)
    print(f'Test MSE: {test_mse:.6f}')
