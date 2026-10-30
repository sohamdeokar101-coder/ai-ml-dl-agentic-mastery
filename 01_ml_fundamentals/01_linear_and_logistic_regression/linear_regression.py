import numpy as np

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using NumPy.
    
    Model:
        y_hat = X @ w + b
        
    Loss (Mean Squared Error):
        J(w, b) = (1 / (2 * m)) * sum((y_hat - y)^2)
        
    Gradients:
        dJ/dw = (1 / m) * X.T @ (y_hat - y)
        dJ/db = (1 / m) * sum(y_hat - y)
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n_features = X.shape
        y = y.reshape(-1, 1)

        # Initialize weights and bias with zeros
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            # 1. Forward pass: compute predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # 2. Compute loss (MSE)
            error = y_pred - y
            loss = (1 / (2 * m)) * np.sum(error ** 2)
            self.loss_history.append(loss)

            # 3. Compute analytical gradients
            dw = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)

            # 4. Gradient descent parameter update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(-1, 1)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-12))
