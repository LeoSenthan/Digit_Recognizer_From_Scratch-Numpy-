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

# SIMPLE LAYER
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # The initialization
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0
    
    def update_params(self, layer):
        self.t += 1
        
        if self.m_weights is None:
            self.m_weights = np.zeros_like(layer.weights)
            self.v_weights = np.zeros_like(layer.weights)
            self.m_biases = np.zeros_like(layer.biases)
            self.v_biases = np.zeros_like(layer.biases)
        
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * layer.dweights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (layer.dweights ** 2)
        
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * layer.dbiases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (layer.dbiases ** 2)
        
        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
        m_hat_biases = self.m_biases / (1 - self.beta1 ** self.t)
        v_hat_biases = self.v_biases / (1 - self.beta2 ** self.t)
        
        layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

def create_digit_model():
    dense1 = Layer_Dense(784, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 32)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(32, 10)
    activation3 = Activation_Softmax()
    
    return dense1, activation1, dense2, activation2, dense3, activation3

def train_model(X_train, y_train, epochs=1000):
    dense1, activation1, dense2, activation2, dense3, activation3 = create_digit_model()
    
    # HIGHER LEARNING RATE
    optimizer1 = Optimizer_Adam(learning_rate=0.01)
    optimizer2 = Optimizer_Adam(learning_rate=0.01)
    optimizer3 = Optimizer_Adam(learning_rate=0.01)
    
    best_accuracy = 0
    best_model = None
    
    
    for epoch in range(epochs):
        # Forward pass
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)
        
        # Calculate loss
        probs = activation3.output
        correct_logprobs = -np.log(np.clip(probs[range(len(X_train)), y_train], 1e-7, 1-1e-7))
        loss = np.mean(correct_logprobs)
        
        # Backward pass
        activation3.backward(probs, y_train)
        dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        # Update parameters
        optimizer1.update_params(dense1)
        optimizer2.update_params(dense2)
        optimizer3.update_params(dense3)
        
        # Calculate accuracy
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y_train)
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (dense1, activation1, dense2, activation2, dense3, activation3)
        
        if epoch % 100 == 0:
            pred_dist = np.bincount(predictions, minlength=10)
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            print(f"  Predictions: {pred_dist}")
        
        # Stop when we get good results
        if accuracy > 0.75:
            print(f"Good accuracy reached at epoch {epoch}")
            break
            
        # Dont train too long if not learning
        if epoch > 200 and accuracy < 0.3:
            print(f"Stopping - not learning after {epoch} epochs")
            break
    
    print(f"Final training accuracy: {accuracy:.4f}")
    return best_model, best_accuracy

def evaluate_model(model, X_test, y_test):
    dense1, activation1, dense2, activation2, dense3, activation3 = model
    
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    probs = activation3.output
    predictions = np.argmax(probs, axis=1)
    test_accuracy = np.mean(predictions == y_test)
    
    return test_accuracy, predictions

if __name__ == "__main__":
    csv_file_path = "digits.csv"
    
    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data(csv_file_path)
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Use the configuration that worked before
        model, train_accuracy = train_model(X_train, y_train, epochs=500)
        
        test_accuracy, predictions = evaluate_model(model, X_test, y_test)
        
        print(f"\n=== RESULTS ===")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        correct_predictions = np.sum(predictions == y_test)
        total_predictions = len(y_test)
        
        print(f"\nTest Set: {correct_predictions}/{total_predictions} correct ({test_accuracy:.1%})")
        
        print(f"\n=== PER-CLASS PERFORMANCE ===")
        for class_id in range(10):
            class_mask = y_test == class_id
            if np.any(class_mask):
                class_correct = np.sum(predictions[class_mask] == class_id)
                class_total = np.sum(class_mask)
                class_accuracy = class_correct / class_total
                print(f"Class {class_id}: {class_accuracy:.1%} ({class_correct}/{class_total})")
        
        print(f"\nPrediction distribution: {np.bincount(predictions, minlength=10)}")
        print(f"Actual distribution: {np.bincount(y_test, minlength=10)}")
        
        print("\n=== DETAILED PREDICTIONS ===")
        for i in range(min(15, len(y_test))):
            actual = y_test[i]
            predicted = predictions[i]
            correct = actual == predicted
            status = "Correct" if correct else "Incorrect"
            print(f"Example {i+1:2d}: Actual={actual}, Predicted={predicted} {status}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()