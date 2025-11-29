from .layers import Layer_Dense
from .activations import Activation_ReLU, Activation_Softmax

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def backward(self, loss_grad, y_true):
        grad = loss_grad
        for layer in reversed(self.layers):
            if isinstance(layer, Activation_Softmax):
                layer.backward(grad, y_true)
            else:
                layer.backward(grad)
            grad = layer.dinputs
        return grad

