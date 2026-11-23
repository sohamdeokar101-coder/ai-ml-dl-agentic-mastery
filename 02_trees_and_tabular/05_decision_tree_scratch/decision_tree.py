import numpy as np
from tree_node import Node
from criterion import compute_information_gain

class DecisionTreeScratch:
    """
    Classification Decision Tree implemented from scratch using the CART algorithm.
    
    Features:
        - Gini Impurity and Shannon Entropy splitting criteria.
        - Pre-pruning via max_depth and min_samples_split.
        - Feature importance tracking (Mean Decrease in Impurity - MDI).
        - ASCII tree structure visualizer.
    """
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini"
    ):
        if criterion not in ["gini", "entropy"]:
            raise ValueError(f"Unknown criterion '{criterion}'. Choose 'gini' or 'entropy'.")
            
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None

    def _should_stop(self, X: np.ndarray, y: np.ndarray, depth: int) -> bool:
        """Determines whether recursion should stop and create a leaf node."""
        if depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if len(np.unique(y)) <= 1:
            return True
        return False

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Finds the optimal feature and numerical threshold that maximizes Information Gain."""
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        m, n_features = X.shape

        for feat in range(n_features):
            values = X[:, feat]
            unique_vals = np.unique(values)

            # Test midpoints between consecutive unique sorted values
            if len(unique_vals) <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gain = compute_information_gain(y, y[left_mask], y[right_mask], criterion=self.criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively builds the binary decision tree."""
        # Check stopping criteria
        if self._should_stop(X, y, depth):
            # Leaf node gets majority class
            classes, counts = np.unique(y, return_counts=True)
            majority_class = classes[np.argmax(counts)]
            return Node(value=majority_class)

        # Find best split
        best_feat, best_thresh, best_gain = self._best_split(X, y)

        # If no split produces positive information gain, become a leaf
        if best_gain <= 1e-7 or best_feat is None:
            classes, counts = np.unique(y, return_counts=True)
            return Node(value=classes[np.argmax(counts)])

        # Accumulate feature importance (MDI)
        self.feature_importances_[best_feat] += best_gain * (len(y) / self.n_total_samples_)

        # Split data and recurse
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            info_gain=best_gain
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the decision tree on training data."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_total_samples_, self.n_features_ = X.shape
        self.feature_importances_ = np.zeros(self.n_features_)

        self.root = self._build_tree(X, y, depth=0)

        # Normalize feature importances so they sum to 1.0
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance

        return self

    def _predict_sample(self, node: Node, x: np.ndarray):
        """Traverses the binary tree for a single sample."""
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for an array of samples."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.array([self._predict_sample(self.root, x) for x in X])

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.array(y_true) == np.array(y_pred)))

    def print_tree(self, node: Node = None, depth: int = 0, feature_names: list = None):
        """Prints a human-readable ASCII representation of the decision rules."""
        if node is None:
            node = self.root

        indent = "|   " * depth

        if node.is_leaf():
            print(f"{indent}|--- Class: {node.value}")
            return

        feat_name = f"Feature {node.feature}" if feature_names is None else feature_names[node.feature]
        print(f"{indent}|--- If {feat_name} <= {node.threshold:.3f}")
        self.print_tree(node.left, depth + 1, feature_names)

        print(f"{indent}|--- If {feat_name} > {node.threshold:.3f}")
        self.print_tree(node.right, depth + 1, feature_names)
