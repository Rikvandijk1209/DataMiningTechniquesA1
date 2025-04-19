import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import optuna
import matplotlib.pyplot as plt


class MoodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_map: dict):
        self.features = df.drop(columns=['id', 'mood', 'date']).values.astype(np.float32)
        self.targets = df['mood'].values.astype(np.float32)  # CHANGED to float32
        self.ids = df['id'].map(id_map).values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            'x': torch.tensor(self.features[idx]),
            'id': torch.tensor(self.ids[idx]),
            'y': torch.tensor(self.targets[idx])
        }
    
class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, id_count: int, id_embed_dim: int):
        super().__init__()
        self.id_embedding = nn.Embedding(id_count, id_embed_dim)
        self.rnn = nn.GRU(input_dim + id_embed_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, 1)  # Output a single float

    def forward(self, x: torch.Tensor, id_tensor: torch.Tensor):
        id_embed = self.id_embedding(id_tensor)
        x_cat = torch.cat([x, id_embed], dim=-1).unsqueeze(1)
        _, h_n = self.rnn(x_cat)
        return self.regressor(h_n.squeeze(0)).squeeze(-1)  # Shape: (batch,)

def train_final_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion, device: torch.device, num_epochs: int = 2000, patience: int = 5):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Train the model on the training set
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate the model on the validation set
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
        
        # If validation loss improves, save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save model parameters
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1
    
    # Load the best model (with the lowest validation loss)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0
    for batch in loader:
        x, id_tensor, y = batch['x'].to(device), batch['id'].to(device), batch['y'].to(device)
        optimizer.zero_grad()
        logits = model(x, id_tensor)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x, id_tensor, y = batch['x'].to(device), batch['id'].to(device), batch['y'].to(device)
            logits = model(x, id_tensor)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def objective(trial: optuna.Trial, train_df: pd.DataFrame, val_df: pd.DataFrame, id_map: dict, input_dim: int, id_count: int, device: torch.device) -> float:
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    id_embed_dim = trial.suggest_int('id_embed_dim', 4, 16)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = RNNRegressor(input_dim, hidden_dim, id_count, id_embed_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(MoodDataset(train_df, id_map), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MoodDataset(val_df, id_map), batch_size=batch_size)

    for epoch in range(10):  # early stop/short training for Optuna
        train_epoch(model, train_loader, optimizer, criterion, device)

    val_loss = evaluate(model, val_loader, criterion, device)
    return val_loss


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_map: dict):
        self.features = df.drop(columns=['id', 'mood', 'date']).values.astype(np.float32)
        self.ids = df['id'].map(id_map).values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            'x': torch.tensor(self.features[idx]),
            'id': torch.tensor(self.ids[idx])
        }

def predict(model: nn.Module, test_df: pd.DataFrame, id_map: dict, device: torch.device, batch_size: int = 32) -> np.ndarray:
    model.eval()
    dataset = TestDataset(test_df, id_map)
    loader = DataLoader(dataset, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in loader:
            x, id_tensor = batch['x'].to(device), batch['id'].to(device)
            outputs = model(x, id_tensor)
            preds.extend(outputs.cpu().numpy())
    return np.array(preds)

def plot_mood_predictions(model, train_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # Disable gradient tracking
        for batch in train_loader:
            inputs = batch['x']  # Using 'x' for inputs
            targets = batch['y']  # Using 'y' for targets
            id_tensor = batch['id']  # Get the 'id' tensor
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            id_tensor = id_tensor.to(device)  # Send the ID tensor to the device
            
            # Make predictions
            outputs = model(inputs, id_tensor)
            predicted = outputs  # raw float predictions
            
            # Store predictions and targets
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Flatten the lists for plotting
    all_preds = [item for sublist in all_preds for item in sublist]
    all_targets = [item for sublist in all_targets for item in sublist]
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')  # Ideal line (y=x)
    plt.xlabel('True Mood')
    plt.ylabel('Predicted Mood')
    plt.title('True vs Predicted Mood')
    plt.show()

def get_accuracy_rate(model, train_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in train_loader:
            inputs = batch['x']
            targets = batch['y']
            id_tensor = batch['id']
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            id_tensor = id_tensor.to(device)
            
            outputs = model(inputs, id_tensor)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy