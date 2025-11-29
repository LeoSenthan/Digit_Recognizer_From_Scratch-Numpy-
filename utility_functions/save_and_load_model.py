import numpy as np

def save_model(nn, filepath="saved_model.npz"):
    """
    Save the weights and biases of all Dense layers in the network.
    
    Args:
        nn: NeuralNetwork instance
        filepath: Path to save the model
    """
    params = {}
    for i, layer in enumerate(nn.layers):
        if hasattr(layer, "weights") and hasattr(layer, "biases"):
            params[f"layer_{i}_weights"] = layer.weights
            params[f"layer_{i}_biases"] = layer.biases
    np.savez(filepath, **params)
    print(f"Model saved to {filepath}")


def load_model(nn, filepath="saved_model.npz"):
    """
    Load weights and biases into the network.
    
    Args:
        nn: NeuralNetwork instance
        filepath: Path to load the model from
    """
    data = np.load(filepath)
    for i, layer in enumerate(nn.layers):
        if hasattr(layer, "weights") and hasattr(layer, "biases"):
            layer.weights = data[f"layer_{i}_weights"]
            layer.biases = data[f"layer_{i}_biases"]
    print(f"Model loaded from {filepath}")
