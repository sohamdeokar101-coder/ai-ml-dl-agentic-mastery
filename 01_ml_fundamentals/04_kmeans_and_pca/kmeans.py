import numpy as np

class KMeansScratch:
    """
    K-Means Clustering implemented from scratch using NumPy.
    
    Algorithm (Lloyd's Expectation-Maximization):
        1. Initialize K centroids (Randomly or K-Means++ style).
        2. Assignment step: Assign each point x^(i) to the closest centroid.
        3. Update step: Recalculate each centroid as the mean of its assigned points.
        4. Repeat until centroids converge (movement < tol) or max_iters reached.
        
    Cost Function (Inertia / Within-Cluster Sum of Squares):
        J = sum_{k=1}^K sum_{x in S_k} ||x - C_k||_2^2
    """
    def __init__(self, n_clusters: int = 3, max_iters: int = 300, tol: float = 1e-4, seed: int = 42):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization to choose well-separated initial centroids."""
        np.random.seed(self.seed)
        m, d = X.shape
        centroids = [X[np.random.choice(m)]]

        for _ in range(1, self.k):
            # Compute distance of each point to the closest existing centroid
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            # Choose next centroid with probability proportional to distance squared
            probs = dist_sq / (np.sum(dist_sq) + 1e-12)
            next_idx = np.random.choice(m, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def fit(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, d = X.shape
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            # 1. Assignment Step (Vectorized broadcasting: (m, 1, d) - (1, K, d) -> (m, K, d))
            diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=2) # shape (m, K)
            self.labels_ = np.argmin(sq_dists, axis=1) # shape (m,)

            # 2. Update Step (Recalculate centroids as mean of assigned points)
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.k):
                cluster_pts = X[self.labels_ == k]
                if len(cluster_pts) > 0:
                    new_centroids[k] = np.mean(cluster_pts, axis=0)
                else:
                    # If empty cluster, reinitialize to a random data point
                    new_centroids[k] = X[np.random.choice(m)]

            # 3. Check for convergence (centroid movement < tolerance)
            centroid_shift = np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break

        # Compute final inertia (Within-Cluster Sum of Squares)
        self.inertia_ = self.compute_inertia(X, self.centroids, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        return np.argmin(sq_dists, axis=1)

    @staticmethod
    def compute_inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
        inertia = 0.0
        for k in range(len(centroids)):
            cluster_pts = X[labels == k]
            if len(cluster_pts) > 0:
                inertia += np.sum((cluster_pts - centroids[k]) ** 2)
        return float(inertia)
