import numpy as np

def gini_impurity(y: np.ndarray) -> float:
    """
    Computes Gini Impurity for a vector of class labels:
        Gini = 1 - sum(p_i^2)
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities ** 2))


def entropy(y: np.ndarray) -> float:
    """
    Computes Shannon Entropy (in bits) for a vector of class labels:
        Entropy = - sum(p_i * log2(p_i))
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Clip probabilities to avoid log2(0) = -inf
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1.0)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_information_gain(
    y_parent: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    criterion: str = "gini"
) -> float:
    """
    Computes Information Gain (reduction in impurity):
        IG = Impurity(Parent) - [ (N_left / N_parent) * Impurity(Left) + (N_right / N_parent) * Impurity(Right) ]
    """
    n_parent = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        return 0.0

    impurity_func = gini_impurity if criterion == "gini" else entropy

    parent_impurity = impurity_func(y_parent)
    weighted_child_impurity = (n_left / n_parent) * impurity_func(y_left) + (n_right / n_parent) * impurity_func(y_right)

    return float(parent_impurity - weighted_child_impurity)
