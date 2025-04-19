import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MoodTabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_map: dict, target_col: str = 'mood'):
        self.features = df.drop(columns=['id', target_col, 'date']).values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.int64)
        self.ids = df['id'].map(id_map).values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'x': self.features[idx],
            'id': self.ids[idx],
            'y': self.targets[idx]
        }


class RandomForestMoodModel:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        report = classification_report(y, preds, output_dict=True)
        return acc, report

    def get_model(self):
        return self.model

    def evaluate_daily_average_performance(self, y_true, y_pred, dates):
        df = pd.DataFrame({'date': dates, 'actual': y_true, 'predicted': y_pred})
        daily_avg = df.groupby('date').mean()

        rmse = np.sqrt(mean_squared_error(daily_avg['actual'], daily_avg['predicted']))
        mean_error = (daily_avg['actual'] - daily_avg['predicted']).mean()

        return rmse, mean_error

def plot_predicted_vs_actual(y_true, y_pred):
    # Rescale the values by dividing by 4
    y_true_rescaled = y_true
    y_pred_rescaled = y_pred

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_rescaled, y_true_rescaled, color='blue', alpha=0.6)
    plt.title('Predicted vs Actual Mood (Rescaled to 0-10)')
    plt.xlabel('Predicted Mood (0-10)')
    plt.ylabel('Actual Mood (0-10)')
    plt.grid(True)
    plt.show()



def split_train_val_per_id(df: pd.DataFrame, val_frac=0.2):
    train_dfs = []
    val_dfs = []
    for uid, group in df.groupby("id"):
        group_sorted = group.sort_values("date")
        split_idx = int((1 - val_frac) * len(group_sorted))
        train_dfs.append(group_sorted.iloc[:split_idx])
        val_dfs.append(group_sorted.iloc[split_idx:])
    return pd.concat(train_dfs), pd.concat(val_dfs)


def train_random_forest(df: pd.DataFrame, id_map: dict, target_col: str = 'mood', **rf_kwargs):
    # Split
    train_df, val_df = split_train_val_per_id(df, val_frac=0.2)

    # Create datasets
    train_ds = MoodTabularDataset(train_df, id_map, target_col)
    val_ds = MoodTabularDataset(val_df, id_map, target_col)

    X_train = np.array([sample['x'] for sample in train_ds])
    y_train = np.array([sample['y'] for sample in train_ds])
    X_val = np.array([sample['x'] for sample in val_ds])
    y_val = np.array([sample['y'] for sample in val_ds])

    # Train model
    model = RandomForestMoodModel(**rf_kwargs)
    model.fit(X_train, y_train)

    # Evaluate
    acc, report = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {acc:.4f}")
    return model, train_ds, val_ds


def predict_test(model: RandomForestMoodModel, df: pd.DataFrame, id_map: dict):
    test_ds = MoodTabularDataset(df, id_map)
    X_test = np.array([sample['x'] for sample in test_ds])
    preds = model.predict(X_test)
    return preds


def plot_rf_predictions(model: RandomForestMoodModel, dataset: MoodTabularDataset):
    X = np.array([sample['x'] for sample in dataset])
    y_true = np.array([sample['y'] for sample in dataset])
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("True Mood")
    plt.ylabel("Predicted Mood")
    plt.title("Random Forest: True vs Predicted Mood")
    plt.grid(True)
    plt.show()

# Function to aggregate classes with few samples into "Other"
def aggregate_classes(labels, min_samples=10):
    """
    Aggregates classes with fewer than `min_samples` occurrences into a new class called "Other".
    """
    class_counts = labels.value_counts()
    small_classes = class_counts[class_counts < min_samples].index.tolist()
    
    # Map classes with fewer than min_samples to 'Other'
    labels_aggregated = labels.replace({cls: 'Other' for cls in small_classes})
    
    return labels_aggregated
