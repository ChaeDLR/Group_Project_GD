import numpy as np


class BatchGDRegressor:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.iterations = None

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        prev_loss = float("inf")

        for i in range(self.epochs):

            # prediction on entire dataset
            y_pred = X @ self.weights + self.bias

            # compute loss
            loss = np.mean((y - y_pred) ** 2)

            # gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # skip comparison on first iteration
            if prev_loss != float("inf"):

                relative_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-8)

                if relative_change < 1e-4:
                    self.iterations = i + 1
                    break

            prev_loss = loss

            # Fallback
        if self.iterations is None:
            self.iterations = self.epochs

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
