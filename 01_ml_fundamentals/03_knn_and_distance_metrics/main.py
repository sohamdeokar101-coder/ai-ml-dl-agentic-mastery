import numpy as np
from knn import KNNClassifierScratch, KNNRegressorScratch
from curse_of_dimensionality import run_curse_of_dimensionality_experiment

def generate_multiclass_blobs(n_samples_per_class: int = 60, seed: int = 42):
    """Generates a 2D 3-class classification problem."""
    np.random.seed(seed)
    c1 = np.random.randn(n_samples_per_class, 2) * 0.7 + np.array([-2.0, -1.0])
    c2 = np.random.randn(n_samples_per_class, 2) * 0.7 + np.array([2.0, 1.0])
    c3 = np.random.randn(n_samples_per_class, 2) * 0.7 + np.array([0.0, 2.5])

    X = np.vstack([c1, c2, c3])
    y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class + [2]*n_samples_per_class)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.25, seed: int = 42):
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    print("=" * 80)
    print("PROJECT 03: K-NEAREST NEIGHBORS (KNN) & CURSE OF DIMENSIONALITY")
    print("=" * 80)

    # 1. Classification Benchmark across Metrics & Weights
    X, y = generate_multiclass_blobs(n_samples_per_class=80, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.3, seed=42)

    print(f"Classification Dataset: {len(X_train)} Train samples, {len(X_test)} Test samples (3 Classes)\n")
    print(f"{'Metric':<12} | {'Weights':<10} | {'K':<4} | {'Test Accuracy (%)':<18}")
    print("-" * 52)

    configs = [
        ("euclidean", "uniform", 3),
        ("euclidean", "distance", 3),
        ("euclidean", "uniform", 7),
        ("manhattan", "uniform", 3),
        ("manhattan", "distance", 3),
        ("cosine", "uniform", 3),
    ]

    for metric, weights, k in configs:
        clf = KNNClassifierScratch(n_neighbors=k, metric=metric, weights=weights)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = clf.accuracy_score(y_test, preds)
        print(f"{metric:<12} | {weights:<10} | {k:<4} | {acc * 100:<18.2f}")

    # 2. Regression Benchmark
    print("\n" + "=" * 80)
    print("PART 2: KNN REGRESSION BENCHMARK")
    print("=" * 80)
    
    np.random.seed(42)
    X_reg_train = np.sort(np.random.uniform(0, 5, size=(80, 1)), axis=0)
    y_reg_train = np.sin(X_reg_train).flatten() + np.random.randn(80) * 0.1
    X_reg_test = np.sort(np.random.uniform(0, 5, size=(40, 1)), axis=0)
    y_reg_test = np.sin(X_reg_test).flatten() + np.random.randn(40) * 0.1

    reg_unif = KNNRegressorScratch(n_neighbors=5, weights="uniform").fit(X_reg_train, y_reg_train)
    reg_dist = KNNRegressorScratch(n_neighbors=5, weights="distance").fit(X_reg_train, y_reg_train)

    mse_unif = KNNRegressorScratch.mse(y_reg_test, reg_unif.predict(X_reg_test))
    mse_dist = KNNRegressorScratch.mse(y_reg_test, reg_dist.predict(X_reg_test))

    print(f"KNN Regressor (K=5, Uniform):  Test MSE = {mse_unif:.4f}")
    print(f"KNN Regressor (K=5, Distance): Test MSE = {mse_dist:.4f}")

    # 3. Curse of Dimensionality Experiment
    print("\n" + "=" * 80)
    print("PART 3: THE CURSE OF DIMENSIONALITY (DISTANCE CONCENTRATION)")
    print("=" * 80)
    
    dim_results = run_curse_of_dimensionality_experiment()
    print(f"{'Dimension (D)':<14} | {'Min Distance':<14} | {'Max Distance':<14} | {'Contrast Ratio ((Max-Min)/Min)'}")
    print("-" * 75)
    for r in dim_results:
        print(f"{r['dimension']:<14} | {r['min_dist']:<14.4f} | {r['max_dist']:<14.4f} | {r['contrast_ratio']:<14.4f}")

    # Generate Visualization Plot
    try:
        import matplotlib.pyplot as plt

        dims = [r["dimension"] for r in dim_results]
        ratios = [r["contrast_ratio"] for r in dim_results]
        min_d = [r["min_dist"] for r in dim_results]
        max_d = [r["max_dist"] for r in dim_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Distance Contrast Ratio collapsing
        ax1.plot(dims, ratios, "o-", color="crimson", linewidth=2, markersize=6)
        ax1.set_title("The Curse of Dimensionality: Distance Contrast Collapses", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Number of Dimensions (D)", fontsize=10)
        ax1.set_ylabel("Contrast Ratio (d_max - d_min) / d_min", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Absolute Min and Max Distances
        ax2.plot(dims, max_d, "s-", color="darkblue", label="Max Distance (d_max)")
        ax2.plot(dims, min_d, "^-", color="darkgreen", label="Min Distance (d_min)")
        ax2.set_title("Absolute Distances Converge Together", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Number of Dimensions (D)", fontsize=10)
        ax2.set_ylabel("Euclidean Distance", fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = "/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/01_ml_fundamentals/03_knn_and_distance_metrics/curse_of_dimensionality.png"
        plt.savefig(plot_path, dpi=150)
        print("\n[SUCCESS] Saved high-dimensional distance analysis plot to 'curse_of_dimensionality.png'")
    except ImportError:
        print("\n[INFO] matplotlib not installed; skipping plot generation.")

if __name__ == "__main__":
    main()
