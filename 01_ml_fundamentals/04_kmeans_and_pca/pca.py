import numpy as np

class PCAScratch:
    """
    Principal Component Analysis (PCA) implemented from scratch using NumPy.
    
    Mathematical Pipeline:
        1. Center data: X_centered = X - mean(X, axis=0)
        2. Covariance matrix: Sigma = (1 / (m - 1)) * (X_centered.T @ X_centered)
        3. Eigendecomposition: Sigma @ v = lambda * v
        4. Sort eigenvalues & eigenvectors in descending order.
        5. Select top k eigenvectors to form projection matrix W of shape (D, k).
        6. Project: X_proj = X_centered @ W
        7. Reconstruction: X_recon = X_proj @ W.T + mean
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None # Projection matrix W (D, k)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, d = X.shape
        if self.n_components > d:
            raise ValueError(f"n_components ({self.n_components}) cannot exceed feature dimension ({d}).")

        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute sample covariance matrix of shape (d, d)
        cov_matrix = (1.0 / (m - 1)) * np.dot(X_centered.T, X_centered)

        # 3. Eigendecomposition (eigh for symmetric covariance matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Store top components and variance metrics
        self.components_ = eigenvectors[:, :self.n_components] # shape (d, k)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCAScratch is not fitted yet! Call fit() before transform().")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_centered = X - self.mean_
        # Project data onto principal axes: (m, d) @ (d, k) -> (m, k)
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_projected: np.ndarray) -> np.ndarray:
        """Reconstructs original feature space from low-dimensional projections."""
        # (m, k) @ (k, d) + (d,) -> (m, d)
        return np.dot(X_projected, self.components_.T) + self.mean_
