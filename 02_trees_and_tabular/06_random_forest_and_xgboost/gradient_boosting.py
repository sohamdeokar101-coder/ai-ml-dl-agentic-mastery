import numpy as np

class RegressionTreeStump:
    """A lightweight regression tree/stump for gradient boosting residual fitting."""
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) < 1e-7:
            self.value = float(np.mean(y))
            return self

        m, n_features = X.shape
        best_variance_reduction = -1.0
        best_feat = None
        best_thresh = None

        parent_var = np.var(y) * len(y)

        for feat in range(n_features):
            values = X[:, feat]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                var_reduction = parent_var - (np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right))

                if var_reduction > best_variance_reduction:
                    best_variance_reduction = var_reduction
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None or best_variance_reduction <= 1e-7:
            self.value = float(np.mean(y))
            return self

        self.feature = best_feat
        self.threshold = best_thresh

        left_mask = X[:, self.feature] <= self.threshold
        right_mask = ~left_mask

        self.left = RegressionTreeStump(self.max_depth, self.min_samples_split).fit(X[left_mask], y[left_mask], depth + 1)
        self.right = RegressionTreeStump(self.max_depth, self.min_samples_split).fit(X[right_mask], y[right_mask], depth + 1)
        return self

    def predict_sample(self, x: np.ndarray) -> float:
        if self.value is not None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict_sample(x)
        else:
            return self.right.predict_sample(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_sample(x) for x in X])


class GradientBoostingRegressorScratch:
    """
    Gradient Boosted Decision Trees Regressor implemented from scratch using NumPy.
    
    Algorithm (Friedman's Gradient Boosting Machine):
        1. Initialize F_0(x) = mean(y)
        2. For m = 1 to M:
            a. Compute pseudo-residuals: r_i = y_i - F_{m-1}(x_i)
            b. Fit a shallow regression tree h_m(x) on residuals r
            c. Update model with learning rate eta: F_m(x) = F_{m-1}(x) + eta * h_m(x)
    """
    def __init__(
        self,
        n_estimators: int = 60,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2
    ):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.base_pred = None
        self.trees = []
        self.train_loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m = len(y)
        self.base_pred = float(np.mean(y))
        current_pred = np.full(m, self.base_pred)

        self.trees = []
        self.train_loss_history = []

        for b in range(self.n_estimators):
            # 1. Compute negative gradient (residuals for MSE loss)
            residuals = y - current_pred

            # Track MSE
            current_mse = float(np.mean(residuals ** 2))
            self.train_loss_history.append(current_mse)

            # 2. Fit a shallow tree to the residuals
            tree = RegressionTreeStump(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # 3. Additive update with shrinkage
            current_pred += self.lr * tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(len(X), self.base_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-12)))
