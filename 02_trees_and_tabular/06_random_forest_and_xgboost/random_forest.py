import numpy as np
import sys
import os

# Import the base DecisionTreeScratch from Project 5
sys.path.append(os.path.abspath("../05_decision_tree_scratch"))
from decision_tree import DecisionTreeScratch

class RandomForestClassifierScratch:
    """
    Random Forest Classifier implemented from scratch using pure NumPy.
    
    Key Mechanisms:
        1. Bootstrap Aggregation (Bagging): Samples N rows with replacement per tree.
        2. Random Feature Subspacing: Selects sqrt(D) features at each node split to de-correlate trees.
        3. Out-Of-Bag (OOB) Validation: Evaluates generalization error using samples excluded from bootstrap.
        4. Majority Voting: Aggregates individual tree predictions for robust ensemble inference.
    """
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        seed: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.oob_score_ = None

    def _get_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            return n_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        m_try = self._get_max_features(n_features)

        self.trees = []
        # Matrix to track which samples were used in which tree (for OOB evaluation)
        in_bag_mask = np.zeros((self.n_estimators, n_samples), dtype=bool)

        for b in range(self.n_estimators):
            # 1. Bootstrap sample (sampling with replacement)
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            in_bag_mask[b, boot_idx] = True

            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # 2. Build decision tree with random feature subspacing
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion="gini"
            )

            # Override _best_split to perform random feature subspacing
            original_best_split = tree._best_split
            def randomized_best_split(X_sub, y_sub, m_feat=m_try, orig=original_best_split):
                # Pick m_try random feature candidates without replacement
                total_f = X_sub.shape[1]
                cand_features = np.random.choice(total_f, size=m_feat, replace=False)
                
                best_gain = -1.0
                best_feat = None
                best_thresh = None

                for feat in cand_features:
                    values = X_sub[:, feat]
                    unique_vals = np.unique(values)
                    if len(unique_vals) <= 1:
                        continue
                    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                    for thresh in thresholds:
                        left_mask = values <= thresh
                        right_mask = ~left_mask
                        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                            continue
                        
                        from criterion import compute_information_gain
                        gain = compute_information_gain(y_sub, y_sub[left_mask], y_sub[right_mask], criterion="gini")
                        if gain > best_gain:
                            best_gain = gain
                            best_feat = feat
                            best_thresh = thresh
                return best_feat, best_thresh, best_gain

            tree._best_split = randomized_best_split
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        # 3. Compute Out-Of-Bag (OOB) Score
        oob_correct = 0
        oob_total = 0

        for i in range(n_samples):
            # Identify trees where sample i was NOT in the bootstrap sample
            oob_tree_indices = np.where(~in_bag_mask[:, i])[0]
            if len(oob_tree_indices) > 0:
                oob_preds = [self.trees[t]._predict_sample(self.trees[t].root, X[i]) for t in oob_tree_indices]
                classes, counts = np.unique(oob_preds, return_counts=True)
                majority_oob = classes[np.argmax(counts)]
                if majority_oob == y[i]:
                    oob_correct += 1
                oob_total += 1

        self.oob_score_ = float(oob_correct / oob_total) if oob_total > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble majority voting across all trees."""
        # Collect predictions from all trees: shape (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        final_preds = []
        for i in range(X.shape[0]):
            sample_votes = tree_preds[:, i]
            classes, counts = np.unique(sample_votes, return_counts=True)
            final_preds.append(classes[np.argmax(counts)])

        return np.array(final_preds)

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.array(y_true) == np.array(y_pred)))
