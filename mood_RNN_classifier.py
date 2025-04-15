import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class MoodRNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MoodRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor):
        # Ensure input x is 3D: (batch_size, seq_len, input_size)
        if len(x.shape) == 2:  # If only batch_size and input_size (seq_len=1)
            x = x.unsqueeze(1)  # Add seq_len dimension: (batch_size, 1, input_size)

        # Pass through RNN
        rnn_out, _ = self.rnn(x)

        # Take the output of the last time step
        last_hidden_state = rnn_out[:, -1, :]  # Last time step's hidden state

        # Pass through final fully connected layer
        out = self.fc_out(last_hidden_state)
        return out


class MoodDataset(Dataset):
    def __init__(self, features: torch.Tensor, target: torch.Tensor):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


class MoodRNNPipeline:
    def __init__(self, id_column: str, time_column: str, target_column: str):
        self.id_column = id_column
        self.time_column = time_column
        self.target_column = target_column
        self.model = None

    def preprocess(self, df: pd.DataFrame, feature_columns: list):
        # Sort by ID and time
        df = df.sort_values(by=[self.id_column, self.time_column])
        
        # Feature normalization (excluding the target)
        feature_df = df[feature_columns]
        scaler = StandardScaler()
        features = scaler.fit_transform(feature_df)

        # Target: mood, don't normalize
        target = df[self.target_column].values

        return features, target

    def create_dataloader(self, features: torch.Tensor, target: torch.Tensor, batch_size: int):
        dataset = MoodDataset(features, target)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 20, learning_rate: float = 1e-3):
        self.model = MoodRNNClassifier(input_size=train_loader.dataset.features.shape[1], hidden_size=32, output_size=1)
        criterion = nn.MSELoss()  # Regression loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1))  # Adjust y_batch shape
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation loop
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch.unsqueeze(1))  # Adjust y_batch shape
                    val_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

    def evaluate(self, test_loader: DataLoader) -> None:
        self.model.eval()
        criterion = nn.MSELoss()  # Since it's a regression task

        all_preds = []
        all_labels = []
        test_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = self.model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1))  # Adjust y_batch shape
                test_loss += loss.item()

                all_preds.append(output.numpy())
                all_labels.append(y_batch.numpy())

        # Compute average test loss
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # Flatten the predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        # Plot Predicted vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(all_labels, all_preds, alpha=0.5, color='blue')
        plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red', linestyle='--')
        plt.xlabel('Actual Mood')
        plt.ylabel('Predicted Mood')
        plt.title('Predicted vs Actual Mood')
        plt.show()
