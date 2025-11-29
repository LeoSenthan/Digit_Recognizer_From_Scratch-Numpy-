import numpy as np

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
