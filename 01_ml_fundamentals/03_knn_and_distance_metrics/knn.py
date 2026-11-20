import numpy as np
from distances import pairwise_distances

class KNNBaseScratch:
    """Base class for K-Nearest Neighbors storing training data and computing distances."""
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        if weights not in ["uniform", "distance"]:
            raise ValueError(f"Unknown weights '{weights}'. Choose from 'uniform', 'distance'.")
        self.k = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Lazy learner: memorize the training samples."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self


class KNNClassifierScratch(KNNBaseScratch):
    """K-Nearest Neighbors Classifier from scratch in pure NumPy."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Compute pairwise distance matrix of shape (N_test, M_train)
        dist_matrix = pairwise_distances(X, self.X_train, metric=self.metric)
        
        predictions = []
        for i in range(len(X)):
            row_dists = dist_matrix[i]
            # Find indices of the K smallest distances
            k_indices = np.argsort(row_dists)[:self.k]
            k_labels = self.y_train[k_indices]
            k_dists = row_dists[k_indices]

            if self.weights == "uniform":
                # Majority vote
                classes, counts = np.unique(k_labels, return_counts=True)
                pred = classes[np.argmax(counts)]
            else:
                # Inverse distance weighted vote
                weights = 1.0 / (k_dists + 1e-5)
                classes = np.unique(k_labels)
                class_scores = {c: 0.0 for c in classes}
                for label, w in zip(k_labels, weights):
                    class_scores[label] += w
                pred = max(class_scores, key=class_scores.get)

            predictions.append(pred)

        return np.array(predictions)

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.array(y_true) == np.array(y_pred)))


class KNNRegressorScratch(KNNBaseScratch):
    """K-Nearest Neighbors Regressor from scratch in pure NumPy."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        dist_matrix = pairwise_distances(X, self.X_train, metric=self.metric)
        
        predictions = []
        for i in range(len(X)):
            row_dists = dist_matrix[i]
            k_indices = np.argsort(row_dists)[:self.k]
            k_targets = self.y_train[k_indices]
            k_dists = row_dists[k_indices]

            if self.weights == "uniform":
                pred = float(np.mean(k_targets))
            else:
                weights = 1.0 / (k_dists + 1e-5)
                pred = float(np.sum(weights * k_targets) / np.sum(weights))

            predictions.append(pred)

        return np.array(predictions)

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
