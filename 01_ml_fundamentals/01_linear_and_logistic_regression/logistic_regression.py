import numpy as np

class LogisticRegressionScratch:
    """
    Binary Logistic Regression implemented from scratch using NumPy.
    
    Model:
        z = X @ w + b
        y_hat = sigmoid(z) = 1 / (1 + exp(-z))
        
    Loss (Binary Cross-Entropy):
        J(w, b) = - (1 / m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))
        
    Gradients:
        dJ/dw = (1 / m) * X.T @ (y_hat - y)
        dJ/db = (1 / m) * sum(y_hat - y)
    """
    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Prevent overflow for large negative z
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n_features = X.shape
        y = y.reshape(-1, 1)

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            # 1. Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # 2. Compute BCE Loss with numerical epsilon clipping
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = - (1 / m) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)

            # 3. Compute gradients
            error = y_pred - y
            dw = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)

            # 4. Parameter update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(-1, 1)
        return float(np.mean(y_true == y_pred))
