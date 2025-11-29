import numpy as np
import pandas as pd

def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def load_and_prepare_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    X = data.drop('label', axis=1).values
    y = data['label'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    class_dist = np.bincount(y)
    print(f"Class distribution: {class_dist}")
    
    X = X / 255.0
    return train_test_split_manual(X, y, test_size=0.2)
