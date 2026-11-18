import numpy as np

def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Expands 1D or 2D feature matrix X into polynomial powers up to `degree`.
    
    Example:
        If X has shape (m, 1) and degree=3:
        Returns matrix with columns [X^1, X^2, X^3] with shape (m, 3).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    powers = [X ** p for p in range(1, degree + 1)]
    return np.hstack(powers)


class StandardScalerScratch:
    """
    Z-Score Feature Scaler implemented from scratch using NumPy.
    
    Formula:
        z = (x - mean) / (std + eps)
        
    Key ML Best Practice:
        - Mean and standard deviation are computed ONLY during fit() on training data.
        - The same training statistics are applied to transform validation/test sets to prevent data leakage.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        # axis=0 computes mean and std per feature column
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScalerScratch is not fitted yet! Call fit() before transform().")
        return (X - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
