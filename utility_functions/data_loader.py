import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Loads the MNIST dataset, flattens images, normalizes pixels,
    and optionally reshuffles and splits into training/test sets.
    
    Returns:
        X_train, X_test, y_train, y_test : np.ndarray
    """
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten 28x28 images to 784-length vectors and normalize
    X_train = X_train.reshape(-1, 28*28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28*28).astype(np.float32) / 255.0

    # Optional: shuffle and re-split to custom test size
    if test_size > 0:
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        np.random.seed(random_state)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_idx = int(len(X) * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of classes: {len(np.unique(y_train))}")
    class_dist = np.bincount(y_train)
    print(f"Class distribution (train): {class_dist}")

    return X_train, X_test, y_train, y_test
