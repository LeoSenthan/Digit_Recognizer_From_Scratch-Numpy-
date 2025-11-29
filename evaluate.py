import numpy as np
from utility_functions.data_loader import load_and_prepare_data
from train import create_digit_model  # should return your NeuralNetwork instance
from utility_functions.save_and_load_model import load_model  # for loading saved weights

def evaluate_model(nn, X_test, y_test):
    probs = nn.forward(X_test)
    predictions = np.argmax(probs, axis=1)
    test_accuracy = np.mean(predictions == y_test)
    return test_accuracy, predictions

def detailed_report(predictions, y_test, num_classes=10):

    print("\n=== PER-CLASS PERFORMANCE ===")
    for class_id in range(num_classes):
        mask = y_test == class_id
        if np.any(mask):
            class_correct = np.sum(predictions[mask] == class_id)
            class_total = np.sum(mask)
            print(f"Class {class_id}: {class_correct}/{class_total} ({class_correct/class_total:.1%})")
    
    print(f"\nPrediction distribution: {np.bincount(predictions, minlength=num_classes)}")
    print(f"Actual distribution: {np.bincount(y_test, minlength=num_classes)}")

if __name__ == "__main__":
    csv_file_path = "data/digits.csv"
    _, X_test, _, y_test = load_and_prepare_data(csv_file_path)  
    
    # Load model
    nn = create_digit_model()
    load_model(nn, "Saved_Model.npz")
    
    # Evaluate
    test_acc, predictions = evaluate_model(nn, X_test, y_test)
    print(f"\n=== RESULTS ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed per-class report
    detailed_report(predictions, y_test)
