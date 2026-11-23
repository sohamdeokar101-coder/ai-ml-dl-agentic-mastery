import numpy as np
from decision_tree import DecisionTreeScratch

def generate_moons(n_samples: int = 300, noise: float = 0.2, seed: int = 42):
    """Generates two interleaving half-circle moons dataset (non-linear)."""
    np.random.seed(seed)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y)
    ]).T
    X += np.random.normal(scale=noise, size=X.shape)
    y = np.hstack([np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)])

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.3, seed: int = 42):
    np.random.seed(seed)
    m = len(X)
    idx = np.random.permutation(m)
    test_size = int(m * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    print("=" * 80)
    print("PROJECT 05: DECISION TREES FROM SCRATCH (CART ALGORITHM)")
    print("=" * 80)

    # 1. Generate non-linear Moons dataset
    X, y = generate_moons(n_samples=300, noise=0.22, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.3, seed=42)
    print(f"Dataset generated: {len(X_train)} Train samples, {len(X_test)} Test samples\n")

    # 2. Benchmarking across depths and criteria
    print("-" * 80)
    print("PART 1: DEPTH & PRUNING BENCHMARK (BIAS-VARIANCE IN TREES)")
    print("-" * 80)
    print(f"{'Model Configuration':<30} | {'Train Acc (%)':<15} | {'Test Acc (%)':<15} | {'Diagnosis'}")
    print("-" * 80)

    configs = [
        ("Depth = 1 (Decision Stump)", 1, "gini"),
        ("Depth = 2 (Shallow Tree)", 2, "gini"),
        ("Depth = 4 (Balanced Tree)", 4, "gini"),
        ("Depth = 4 (Entropy Criterion)", 4, "entropy"),
        ("Depth = 15 (Unconstrained)", 15, "gini"),
    ]

    trained_models = []

    for name, depth, crit in configs:
        tree = DecisionTreeScratch(max_depth=depth, min_samples_split=2, criterion=crit)
        tree.fit(X_train, y_train)

        train_acc = DecisionTreeScratch.accuracy(y_train, tree.predict(X_train)) * 100
        test_acc = DecisionTreeScratch.accuracy(y_test, tree.predict(X_test)) * 100

        if depth <= 2:
            diag = "Underfitting (High Bias)"
        elif train_acc > 98.0 and (train_acc - test_acc) > 8.0:
            diag = "Overfitting (High Variance)"
        else:
            diag = "Optimal Generalization"

        print(f"{name:<30} | {train_acc:<15.2f} | {test_acc:<15.2f} | {diag}")
        trained_models.append((name, tree, depth))

    # 3. Print ASCII Tree of the Balanced Model
    print("\n" + "=" * 80)
    print("PART 2: INTERPRETABLE ASCII DECISION TREE (Depth = 3 Example)")
    print("=" * 80)
    tree_interp = DecisionTreeScratch(max_depth=3, criterion="gini").fit(X_train, y_train)
    tree_interp.print_tree(feature_names=["Feature_X0 (Horizontal)", "Feature_X1 (Vertical)"])

    # 4. Feature Importances (MDI)
    print("\n" + "-" * 80)
    print("PART 3: FEATURE IMPORTANCES (MEAN DECREASE IN IMPURITY - MDI)")
    print("-" * 80)
    tree_balanced = trained_models[2][1]
    print(f"Feature_X0 Importance: {tree_balanced.feature_importances_[0] * 100:.2f}%")
    print(f"Feature_X1 Importance: {tree_balanced.feature_importances_[1] * 100:.2f}%")

    # 5. Generate Visual Decision Boundary Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plot_configs = [
            (trained_models[0][1], "Underfitting: Depth = 1", axes[0]),
            (trained_models[2][1], "Optimal: Depth = 4", axes[1]),
            (trained_models[4][1], "Overfitting: Depth = 15", axes[2]),
        ]

        # Create mesh grid for boundary visualization
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        for model, title, ax in plot_configs:
            Z = model.predict(grid_points).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors="k", s=30, alpha=0.8)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Feature X0")
            ax.set_ylabel("Feature X1")
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plot_path = "/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/02_trees_and_tabular/05_decision_tree_scratch/decision_tree_boundary.png"
        plt.savefig(plot_path, dpi=150)
        print("\n[SUCCESS] Saved Decision Boundary comparison plot to 'decision_tree_boundary.png'")

    except ImportError:
        print("\n[INFO] matplotlib not installed; skipping plot generation.")

if __name__ == "__main__":
    main()
