from models import GNN
from utilities import prepare_graph_data, process_simulation_data, ParticleDataset, collate_fn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

import os
from glob import glob
from typing import Optional, List, Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model: nn.Module, 
          train_simulation_list: List, 
          test_simulation_list: List,
          n_epochs: Optional[int] = 100, 
          lr: Optional[float] = 5e-4, 
          device: Optional[str | torch.device] = 'cpu',
          checkpoint_dir: Optional[str] = 'checkpoints',
          checkpoint_every: Optional[int] = 10) -> Dict:
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
        Learning rate of Adam optimiser, default is 5e-4.
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

    data_pairs_train = process_simulation_data(train_simulation_list)
    data_pairs_test = process_simulation_data(test_simulation_list)
    
    train_dataset = ParticleDataset(data_pairs_train)
    test_dataset = ParticleDataset(data_pairs_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
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
            for x, y in batch:
                x = x.to(device)
                y = y.to(device)
                
                h, edge_index, edge_attr = prepare_graph_data(x, device=device)
                
                predictions = model(h, edge_index, edge_attr)
                
                y = y.transpose(0, 1)  # From (3, N) to (N, 3)
                
                loss = criterion(predictions, y)
                batch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Average loss over samples in batch (though batch_size=1 here)
            batch_loss = batch_loss / len(batch)
            train_losses.append(batch_loss)
            
            # Update progress bar
            pbar.set_postfix({'train_loss': f'{batch_loss:.6f}'})
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0
                for x, y in batch:
                    x = x.to(device)
                    y = y.to(device)
                    
                    h, edge_index, edge_attr = prepare_graph_data(x, device=device)
                    predictions = model(h, edge_index, edge_attr)
                    
                    y = y.transpose(0, 1)  # From (3, N) to (N, 3)
                    loss = criterion(predictions, y)
                    batch_loss += loss.item()
                
                batch_loss = batch_loss / len(batch)
                val_losses.append(batch_loss)
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, 'best_model.pt')
        
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
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print('-' * 30)
    
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
    n_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            for x, y in batch:
                x = x.to(device)
                y = y.to(device)
                
                h, edge_index, edge_attr = prepare_graph_data(x, device=device)
                predictions = model(h, edge_index, edge_attr)
                
                y = y.transpose(0, 1)  # From (3, N) to (N, 3)
                loss = nn.MSELoss()(predictions, y)
                
                total_loss += loss.item()
                n_samples += 1
    
    return total_loss / n_samples

if __name__ == '__main__':

    train_glob = glob('./data/simulation_train_*')
    train_simulations = [np.load(sim) for sim in train_glob]

    test_glob = glob('./data/simulation_test_*')
    test_simulations = [np.load(sim) for sim in test_glob]

    model = GNN(
        n_layers=3,
        in_node_nf=3,
        in_edge_nf=0,
        hidden_nf=64,
        device=device
    ).double()

    history = train(
        model=model,
        train_simulation_list=train_simulations,
        test_simulation_list=test_simulations,
        n_epochs=10,
        lr=5e-4,
        device=device
    )

    test_mse = evaluate_model(model, test_simulations, device=device)
    print(f'Test MSE: {test_mse:.6f}')
