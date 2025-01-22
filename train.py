from models import GNN
from utilities import prepare_graph_data, process_simulation_data, ParticleDataset, collate_fn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, simulation_list, 
              n_epochs=100, 
              lr=1e-3, 
              device='cpu',
              val_split=0.1):
    """
    Train the GNN model on particle simulation data with variable N.
    
    Args:
        model: GNN model instance
        simulation_list: List of simulation arrays (timesteps, 3, N)
        n_epochs: Number of epochs to train
        lr: Learning rate
        device: torch device
        val_split: Fraction of data to use for validation
    
    Returns:
        dict: Training history
    """
    # Process data into pairs
    data_pairs = process_simulation_data(simulation_list)
    
    # Create dataset
    dataset = ParticleDataset(data_pairs)
    
    # Split dataset
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Must use batch_size=1 for variable N
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=collate_fn
    )
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            batch_loss = 0
            # Each batch contains a single (x, y) pair due to batch_size=1
            for x, y in batch:
                x = x.to(device)
                y = y.to(device)
                
                # Prepare graph data
                h, edge_index, edge_attr = prepare_graph_data(x, device=device)
                
                # Forward pass
                predictions = model(h, edge_index, edge_attr)
                
                # Reshape predictions and target to match
                n_particles = x.size(1)
                y = y.transpose(0, 1)  # From (3, N) to (N, 3)
                
                # Compute loss
                loss = criterion(predictions, y)
                batch_loss += loss.item()
                
                # Backward pass and optimization
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
        
        # Compute average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, 'best_model.pt')
        
        # Print progress
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print('-' * 30)
    
    return history

def evaluate_model(model, simulation_list, device='cpu'):
    """
    Evaluate the model on test data with variable N.
    
    Args:
        model: Trained GNN model
        simulation_list: List of test simulation arrays
        device: torch device
    
    Returns:
        float: Mean squared error on test data
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
    # Initialize model
    model = GNN(
        n_layers=3,
        in_node_nf=3,  # 3D positions
        in_edge_nf=0,  # No edge features for now
        hidden_nf=64,
        device=device
    ).double()

    # Train model
    history = train(
        model=model,
        simulation_list=train_simulations,
        n_epochs=10,
        lr=5e-4,
        device=device
    )

    # Evaluate on test data
    test_mse = evaluate_model(model, test_simulations, device=device)
    print(f'Test MSE: {test_mse:.6f}')
