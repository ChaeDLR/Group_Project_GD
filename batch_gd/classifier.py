import numpy as np


def sigmoid(z):
    """Takes a real-valued input and maps it to a number between the (0, 1) interval"""
    return 1 / (1 + np.exp(-z))


class BatchGDClassifier:
    weights = None
    bias = None

    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: numpy array of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = self.predict_probability(X)

            # compute gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
    def predict_probability(self, X: np.ndarray):
        z = X @ self.weights + self.bias
        return sigmoid(z)
    
    def predict(self, X: np.ndarray):
        probabilities = self.predict_probability(X)
        return (probabilities >= 0.5).astype(int)  
    
    def score(self, X, y):
        y_preds = self.predict(X)
        return np.mean(y_preds == y)
