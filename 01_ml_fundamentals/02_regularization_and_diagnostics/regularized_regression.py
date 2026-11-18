import numpy as np

class RegularizedRegressionScratch:
    """
    Linear / Polynomial Regression with L1 (Lasso) and L2 (Ridge) Regularization from scratch in NumPy.
    
    Model:
        y_hat = X @ w + b
        
    Loss Functions:
        - No reg:    MSE = (1 / (2 * m)) * sum((y_hat - y)^2)
        - L2 (Ridge): MSE + (lambda / (2 * m)) * sum(w^2)
        - L1 (Lasso): MSE + (lambda / m) * sum(|w|)
        
    Gradients:
        - dJ/dw (base) = (1 / m) * X.T @ (y_hat - y)
        - dJ/dw (L2)   = dJ/dw (base) + (lambda / m) * w
        - dJ/dw (L1)   = dJ/dw (base) + (lambda / m) * sign(w)
        - dJ/db        = (1 / m) * sum(y_hat - y)   [Never regularize the bias!]
    """
    def __init__(
        self,
        penalty: str = "none",
        lambda_reg: float = 0.1,
        learning_rate: float = 0.05,
        n_iterations: int = 1500
    ):
        if penalty not in ["none", "l1", "l2"]:
            raise ValueError(f"Unknown penalty '{penalty}'. Choose from 'none', 'l1', 'l2'.")
            
        self.penalty = penalty
        self.lambda_reg = lambda_reg
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.train_loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        m, n_features = X.shape
        y = y.reshape(-1, 1)

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.train_loss_history = []

        for _ in range(self.n_iterations):
            # 1. Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # 2. Compute Loss for monitoring
            mse_loss = (1 / (2 * m)) * np.sum(error ** 2)
            reg_loss = 0.0
            if self.penalty == "l2":
                reg_loss = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
            elif self.penalty == "l1":
                reg_loss = (self.lambda_reg / m) * np.sum(np.abs(self.weights))
                
            total_loss = mse_loss + reg_loss
            self.train_loss_history.append(total_loss)

            # 3. Base MSE Gradients
            dw = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)

            # 4. Add Regularization Penalty to dw (never to db!)
            if self.penalty == "l2":
                dw += (self.lambda_reg / m) * self.weights
            elif self.penalty == "l1":
                dw += (self.lambda_reg / m) * np.sign(self.weights)

            # 5. Gradient Descent Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Optional Soft-Thresholding for exact Lasso sparsity at zero:
            if self.penalty == "l1":
                threshold = (self.lr * self.lambda_reg) / m
                self.weights[np.abs(self.weights) < threshold] = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(-1, 1)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(-1, 1)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - (ss_res / (ss_tot + 1e-12)))
