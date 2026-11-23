import numpy as np
import sys
import os

sys.path.append(os.path.abspath("../05_decision_tree_scratch"))
from decision_tree import DecisionTreeScratch
from random_forest import RandomForestClassifierScratch
from gradient_boosting import GradientBoostingRegressorScratch

def generate_classification_data(n_samples: int = 300, noise: float = 0.25, seed: int = 42):
    """Generates a noisy interleaving dataset with multiple redundant features (D=6)."""
    np.random.seed(seed)
    n_half = n_samples // 2

    # Two circular patterns
    r1 = np.random.uniform(0.1, 1.2, n_half)
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)

    r2 = np.random.uniform(1.8, 3.0, n_half)
    theta2 = np.random.uniform(0, 2*np.pi, n_half)
    X2 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)

    X_core = np.vstack([X1, X2])
    y = np.array([0]*n_half + [1]*n_half)

    # Add 4 noisy/redundant features to test random feature subspacing (Total D = 6)
    noise_features = np.random.randn(n_samples, 4) * 0.8
    X = np.hstack([X_core, noise_features])

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.3, seed: int = 42):
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    print("=" * 80)
    print("PROJECT 06: RANDOM FORESTS & GRADIENT BOOSTING PIPELINE")
    print("=" * 80)

    # 1. Classification Benchmark: Single Tree vs Random Forest
    X_cls, y_cls = generate_classification_data(n_samples=300, noise=0.25, seed=42)
    X_tr_cls, y_tr_cls, X_te_cls, y_te_cls = train_test_split(X_cls, y_cls, test_ratio=0.3, seed=42)

    print(f"Classification Dataset: {len(X_tr_cls)} Train, {len(X_te_cls)} Test samples across {X_cls.shape[1]} features\n")
    print("-" * 80)
    print("PART 1: BAGGING BENCHMARK (SINGLE TREE VS. RANDOM FOREST)")
    print("-" * 80)

    # Single Decision Tree
    single_tree = DecisionTreeScratch(max_depth=10, min_samples_split=2)
    single_tree.fit(X_tr_cls, y_tr_cls)
    tr_acc_tree = DecisionTreeScratch.accuracy(y_tr_cls, single_tree.predict(X_tr_cls)) * 100
    te_acc_tree = DecisionTreeScratch.accuracy(y_te_cls, single_tree.predict(X_te_cls)) * 100

    # Random Forest (50 Trees with sqrt(D) subspacing)
    rf = RandomForestClassifierScratch(n_estimators=40, max_depth=8, max_features="sqrt", seed=42)
    rf.fit(X_tr_cls, y_tr_cls)
    tr_acc_rf = rf.accuracy(y_tr_cls, rf.predict(X_tr_cls)) * 100
    te_acc_rf = rf.accuracy(y_te_cls, rf.predict(X_te_cls)) * 100

    print(f"{'Model':<30} | {'Train Acc (%)':<15} | {'Test Acc (%)':<15} | {'OOB Score (%)':<15}")
    print("-" * 80)
    print(f"{'Single Decision Tree':<30} | {tr_acc_tree:<15.2f} | {te_acc_tree:<15.2f} | {'N/A':<15}")
    print(f"{'Random Forest (40 Trees)':<30} | {tr_acc_rf:<15.2f} | {te_acc_rf:<15.2f} | {rf.oob_score_*100:<15.2f}")
    print("\n[INSIGHT] Notice how the OOB Score closely mirrors the Test Accuracy without needing a test split!")

    # 2. Regression Benchmark: Gradient Boosting
    print("\n" + "-" * 80)
    print("PART 2: BOOSTING BENCHMARK (GRADIENT BOOSTED DECISION TREES)")
    print("-" * 80)

    np.random.seed(42)
    X_reg = np.sort(np.random.uniform(-3, 3, size=(200, 1)), axis=0)
    y_reg = np.sin(X_reg).flatten() * 2.0 + (X_reg.flatten() ** 2) * 0.3 + np.random.randn(200) * 0.25

    X_tr_reg, y_tr_reg, X_te_reg, y_te_reg = train_test_split(X_reg, y_reg, test_ratio=0.3, seed=42)

    gbr = GradientBoostingRegressorScratch(n_estimators=60, learning_rate=0.1, max_depth=3)
    gbr.fit(X_tr_reg, y_tr_reg)

    y_te_pred = gbr.predict(X_te_reg)
    test_mse = GradientBoostingRegressorScratch.mse(y_te_reg, y_te_pred)
    test_r2 = GradientBoostingRegressorScratch.r2_score(y_te_reg, y_te_pred)

    print(f"Gradient Boosting Trees: {gbr.n_estimators} Estimators, Learning Rate (eta) = {gbr.lr}")
    print(f"Initial Baseline Loss (F0): {gbr.train_loss_history[0]:.4f} MSE")
    print(f"Final Ensemble Loss (F60):  {gbr.train_loss_history[-1]:.4f} MSE")
    print(f"Test Set Performance:       MSE = {test_mse:.4f} | R^2 Score = {test_r2*100:.2f}%")

    # 3. Generate Visual Plots
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Loss Reduction in Gradient Boosting
        ax1.plot(range(1, gbr.n_estimators + 1), gbr.train_loss_history, "o-", color="crimson", linewidth=2, markersize=4)
        ax1.set_title("Gradient Boosting: Residual Loss Decay Across Iterations", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Boosting Round (Trees)", fontsize=10)
        ax1.set_ylabel("Mean Squared Error (Residual Loss)", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Gradient Boosting Fitted Curve on Test Data
        X_dense = np.linspace(-3, 3, 300).reshape(-1, 1)
        y_dense = gbr.predict(X_dense)

        ax2.scatter(X_tr_reg, y_tr_reg, color="blue", alpha=0.5, label="Train Points", s=25)
        ax2.scatter(X_te_reg, y_te_reg, color="green", alpha=0.7, label="Test Points", s=30)
        ax2.plot(X_dense, y_dense, color="red", linewidth=2.5, label=f"GBR Fit (R^2 = {test_r2*100:.1f}%)")
        ax2.set_title("Gradient Boosting: Sequential Additive Curve Fit", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Feature X", fontsize=10)
        ax2.set_ylabel("Target Y", fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = "/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/02_trees_and_tabular/06_random_forest_and_xgboost/ensemble_benchmark.png"
        plt.savefig(plot_path, dpi=150)
        print("\n[SUCCESS] Saved ensemble comparison plot to 'ensemble_benchmark.png'")

    except ImportError:
        print("\n[INFO] matplotlib not installed; skipping plot generation.")

if __name__ == "__main__":
    main()
