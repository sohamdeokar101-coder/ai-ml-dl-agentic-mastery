import itertools
import math
import numpy as np


class ShapleyExplainerScratch:
    """
    Model-agnostic Shapley Value (SHAP) explainer implemented from scratch in NumPy.

    Supports:
        1. Exact Shapley Values (O(2^D)) for low-dimensional tabular models,
           guaranteeing exact Local Accuracy: phi_0 + sum(phi_i) == f(x).
        2. Monte Carlo Permutation Shapley (O(M * D)) for scalable approximation.
    """

    def __init__(self, predict_fn, background_X: np.ndarray):
        self.predict_fn = predict_fn
        self.background_X = np.array(background_X, dtype=float)
        self.expected_value_ = float(np.mean(self.predict_fn(self.background_X)))

    @staticmethod
    def shapley_weight(n_features: int, subset_size: int) -> float:
        """Combinatorial Shapley weight: |S|! * (D - |S| - 1)! / D!"""
        num = math.factorial(subset_size) * math.factorial(n_features - subset_size - 1)
        den = math.factorial(n_features)
        return num / den

    def _coalition_value(self, x: np.ndarray, coalition: tuple) -> float:
        """
        Computes v(S) = E[f(X) | X_S = x_S] by replacing absent features
        with values from the background dataset and averaging predictions.
        """
        X_masked = self.background_X.copy()
        if len(coalition) > 0:
            X_masked[:, list(coalition)] = x[list(coalition)]
        return float(np.mean(self.predict_fn(X_masked)))

    def explain_exact(self, x: np.ndarray) -> np.ndarray:
        """Computes exact Shapley values across all 2^(D-1) coalitions per feature."""
        x = np.asarray(x, dtype=float).flatten()
        n_features = len(x)
        all_features = set(range(n_features))
        shap_values = np.zeros(n_features)

        for i in range(n_features):
            other_features = sorted(all_features - {i})
            for size in range(len(other_features) + 1):
                weight = self.shapley_weight(n_features, size)
                for subset in itertools.combinations(other_features, size):
                    v_without = self._coalition_value(x, subset)
                    v_with = self._coalition_value(x, subset + (i,))
                    shap_values[i] += weight * (v_with - v_without)

        return shap_values

    def explain_monte_carlo(self, x: np.ndarray, n_samples: int = 200, seed: int = 42) -> np.ndarray:
        """Approximates Shapley values using Monte Carlo permutation sampling."""
        rng = np.random.default_rng(seed)
        x = np.asarray(x, dtype=float).flatten()
        n_features = len(x)
        n_bg = len(self.background_X)
        shap_values = np.zeros(n_features)

        for i in range(n_features):
            marginal_sum = 0.0
            for _ in range(n_samples):
                perm = rng.permutation(n_features)
                z = self.background_X[rng.integers(0, n_bg)]
                idx_of_i = int(np.where(perm == i)[0][0])

                x_with_i = z.copy()
                x_without_i = z.copy()

                before_features = perm[:idx_of_i]
                x_with_i[before_features] = x[before_features]
                x_without_i[before_features] = x[before_features]
                x_with_i[i] = x[i]

                batch = np.vstack([x_with_i, x_without_i])
                preds = self.predict_fn(batch)
                marginal_sum += float(preds[0] - preds[1])

            shap_values[i] = marginal_sum / n_samples

        return shap_values


class LimeExplainerScratch:
    """
    Local Interpretable Model-agnostic Explanations (LIME) from scratch in NumPy.

    Generates Gaussian perturbations around target x*, weights them with an
    exponential RBF kernel pi(z) = exp(-||z - x*||^2 / sigma^2), and fits a
    closed-form Weighted Ridge Regression surrogate:
        w* = (Z^T W Z + lambda * I)^(-1) Z^T W f(Z)
    """

    def __init__(self, predict_fn, training_X: np.ndarray, kernel_width: float = 1.5, reg_lambda: float = 1e-2):
        self.predict_fn = predict_fn
        self.feature_stds_ = np.std(training_X, axis=0) + 1e-8
        self.kernel_width = kernel_width
        self.reg_lambda = reg_lambda

    def explain(self, x: np.ndarray, n_perturbations: int = 1000, seed: int = 42):
        rng = np.random.default_rng(seed)
        x = np.asarray(x, dtype=float).flatten()
        n_features = len(x)

        # 1. Perturb around x* scaled by feature standard deviations
        noise = rng.normal(0.0, 0.5, size=(n_perturbations, n_features)) * self.feature_stds_
        Z = x + noise
        Z[0] = x  # Include the exact target instance

        # 2. Query black-box model
        y_blackbox = self.predict_fn(Z)

        # 3. Compute standardized Euclidean distances & exponential proximity weights
        scaled_diff = (Z - x) / self.feature_stds_
        distances = np.sqrt(np.sum(scaled_diff ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # 4. Fit Weighted Ridge Regression on centered perturbations (Z - x) with intercept
        Z_centered = np.hstack([np.ones((n_perturbations, 1)), scaled_diff])
        W = weights[:, np.newaxis]
        Z_weighted = Z_centered * W

        # Normal equation: (Z^T W Z + lambda * I) theta = Z^T W y
        reg_matrix = self.reg_lambda * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0  # Do not regularize intercept

        A = Z_centered.T @ Z_weighted + reg_matrix
        b = Z_weighted.T @ y_blackbox
        theta = np.linalg.solve(A, b)

        local_intercept = float(theta[0])
        local_weights = theta[1:]
        return local_intercept, local_weights


def permutation_feature_importance(
    predict_fn,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric_fn,
    n_repeats: int = 10,
    seed: int = 42
):
    """
    Computes global Permutation Feature Importance on held-out validation data.
    Returns (mean_importances, std_importances).
    """
    rng = np.random.default_rng(seed)
    baseline_score = float(metric_fn(y_val, predict_fn(X_val)))
    n_features = X_val.shape[1]

    mean_importances = np.zeros(n_features)
    std_importances = np.zeros(n_features)

    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_corrupted = X_val.copy()
            X_corrupted[:, j] = rng.permutation(X_corrupted[:, j])
            corrupted_score = float(metric_fn(y_val, predict_fn(X_corrupted)))
            drops.append(baseline_score - corrupted_score)

        mean_importances[j] = float(np.mean(drops))
        std_importances[j] = float(np.std(drops))

    return mean_importances, std_importances
