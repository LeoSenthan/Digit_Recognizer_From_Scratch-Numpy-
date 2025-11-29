import numpy as np
from utility_functions.data_loader import load_and_prepare_data
from models.layers import Layer_Dense
from models.activations import Activation_ReLU, Activation_Softmax
from models.network import NeuralNetwork
from models.optimizers import Optimizer_Adam
from utility_functions.save_and_load_model import save_model

def create_digit_model():
    nn = NeuralNetwork()
    nn.add(Layer_Dense(784, 64))
    nn.add(Activation_ReLU())
    nn.add(Layer_Dense(64, 32))
    nn.add(Activation_ReLU())
    nn.add(Layer_Dense(32, 10))
    nn.add(Activation_Softmax())
    return nn

def train_model(nn, X_train, y_train, epochs=500, learning_rate=0.01):
    optimizers = [Optimizer_Adam(learning_rate) for layer in nn.layers if isinstance(layer, Layer_Dense)]

    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        # Forward pass
        output = nn.forward(X_train)

        # Compute loss (categorical cross-entropy)
        probs = output
        correct_logprobs = -np.log(np.clip(probs[np.arange(len(y_train)), y_train], 1e-7, 1-1e-7))
        loss = np.mean(correct_logprobs)

        # Backward pass
        nn.backward(probs, y_train)

        # Update Dense layers
        optimizer_idx = 0
        for layer in nn.layers:
            if isinstance(layer, Layer_Dense):
                optimizers[optimizer_idx].update_params(layer)
                optimizer_idx += 1

        # Accuracy
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y_train)

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = nn

        # Logging every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            pred_dist = np.bincount(predictions, minlength=10)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, Pred Dist={pred_dist}")

        # Early stopping conditions
        if accuracy > 0.95:
            print(f"Reached 95% training accuracy at epoch {epoch}, stopping early.")
            break
        if epoch > 100 and accuracy < 0.2:
            print(f"Low accuracy after {epoch} epochs, stopping early.")
            break

    print(f"Final training accuracy: {best_accuracy:.4f}")
    return best_model, best_accuracy


if __name__ == "__main__":
    csv_file_path = "data/digits.csv"
    X_train, X_test, y_train, y_test = load_and_prepare_data(csv_file_path)
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Create and train model
    nn = create_digit_model()
    nn, train_acc = train_model(nn, X_train, y_train, epochs=500, learning_rate=0.01)
    save_model(nn,"Saved_Model.npz")
