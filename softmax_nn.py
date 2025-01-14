import numpy as np

# Define activation functions
def softmax(logits):
    """
    Softmax function for multi-class classification.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability improvement
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def relu(z):
    """
    ReLU activation function.
    """
    return np.maximum(0, z)

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """
        Perform the forward pass through the network.
        """
        # Layer 1: Hidden layer with ReLU activation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        
        # Layer 2: Output layer with softmax activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        
        return self.A2  # Output probabilities
    
    def compute_loss(self, Y, Y_hat):
        """
        Compute the loss using cross-entropy.
        """
        m = Y.shape[0]
        log_likelihood = -np.log(Y_hat[range(m), Y])
        loss = np.sum(log_likelihood) / m
        return loss

# Example usage:
if __name__ == "__main__":
    # Initialize the neural network with 3 input features, 5 hidden neurons, and 3 output classes
    nn = NeuralNetwork(input_size=3, hidden_size=5, output_size=3)
    
    # Input data (5 samples, 3 features)
    X = np.array([[1.0, 2.0, 3.0],
                  [0.5, 1.5, 2.5],
                  [1.5, 2.5, 3.5],
                  [2.0, 1.0, 0.5],
                  [3.0, 3.0, 3.0]])  # More input samples added

    # True labels (class indices)
    Y = np.array([0, 1, 2, 1, 0])  # Example true labels (5 samples)

    # Forward pass to compute softmax output
    Y_hat = nn.forward(X)
    
    # Calculate loss
    loss = nn.compute_loss(Y, Y_hat)
    
    print("Softmax Output (Probabilities):\n", Y_hat)
    print("Cross-entropy Loss:", loss)
