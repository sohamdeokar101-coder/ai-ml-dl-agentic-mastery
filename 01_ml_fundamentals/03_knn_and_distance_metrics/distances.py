import numpy as np

def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Computes Euclidean (L2) distance between two 1D vectors."""
    return float(np.sqrt(np.sum((u - v) ** 2)))


def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Computes Manhattan (L1) distance between two 1D vectors."""
    return float(np.sum(np.abs(u - v)))


def cosine_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    """Computes Cosine distance (1 - cosine similarity) between two 1D vectors."""
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    similarity = dot_product / (norm_u * norm_v + eps)
    return float(1.0 - similarity)


def pairwise_distances(X_test: np.ndarray, X_train: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Computes all pairwise distances between X_test of shape (N, D) and X_train of shape (M, D).
    Returns a matrix of shape (N, M) where element [i, j] is the distance between X_test[i] and X_train[j].
    
    Uses fully vectorized NumPy operations (no Python nested loops).
    """
    if metric == "euclidean":
        # Broadcasting: (N, 1, D) - (1, M, D) -> (N, M, D)
        diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    elif metric == "manhattan":
        diff = np.abs(X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :])
        return np.sum(diff, axis=2)

    elif metric == "cosine":
        # Normalize each row to unit length
        eps = 1e-9
        X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + eps)
        X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + eps)
        # Cosine similarity matrix = X_test_norm @ X_train_norm.T (shape: N, M)
        sim = np.dot(X_test_norm, X_train_norm.T)
        return 1.0 - sim

    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from 'euclidean', 'manhattan', 'cosine'.")
