import numpy as np
from kmeans import KMeansScratch
from pca import PCAScratch

def generate_high_dimensional_clusters(n_samples_per_cluster: int = 100, n_features: int = 8, seed: int = 42):
    """Generates synthetic high-dimensional data with 3 true underlying clusters."""
    np.random.seed(seed)
    
    # Define 3 cluster centers in 8D space
    c1 = np.ones(n_features) * -2.5
    c2 = np.ones(n_features) * 0.0
    c3 = np.ones(n_features) * 3.0

    X1 = np.random.randn(n_samples_per_cluster, n_features) * 0.9 + c1
    X2 = np.random.randn(n_samples_per_cluster, n_features) * 1.1 + c2
    X3 = np.random.randn(n_samples_per_cluster, n_features) * 0.8 + c3

    X = np.vstack([X1, X2, X3])
    y_true = np.array([0]*n_samples_per_cluster + [1]*n_samples_per_cluster + [2]*n_samples_per_cluster)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y_true[idx]

def main():
    print("=" * 80)
    print("PROJECT 04: UNSUPERVISED LEARNING — K-MEANS & PCA FROM SCRATCH")
    print("=" * 80)

    # 1. Generate 8-Dimensional Dataset
    X, y_true = generate_high_dimensional_clusters(n_samples_per_cluster=100, n_features=8, seed=42)
    print(f"Generated Dataset: {X.shape[0]} samples with {X.shape[1]} features (3 natural clusters)\n")

    # 2. Part 1: K-Means Clustering & The Elbow Method
    print("-" * 80)
    print("PART 1: K-MEANS ELBOW METHOD (FINDING OPTIMAL K)")
    print("-" * 80)
    print(f"{'K (Clusters)':<15} | {'Inertia (WCSS)':<20} | {'Delta Inertia (Drop)':<20}")
    print("-" * 60)

    k_values = list(range(1, 8))
    inertias = []

    prev_inertia = None
    for k in k_values:
        km = KMeansScratch(n_clusters=k, max_iters=200, seed=42)
        km.fit(X)
        inertias.append(km.inertia_)
        
        drop_str = f"{- (km.inertia_ - prev_inertia):.2f}" if prev_inertia is not None else "---"
        print(f"{k:<15} | {km.inertia_:<20.2f} | {drop_str:<20}")
        prev_inertia = km.inertia_

    # 3. Part 2: Principal Component Analysis (PCA)
    print("\n" + "-" * 80)
    print("PART 2: PCA DIMENSIONALITY REDUCTION (8D -> 2D)")
    print("-" * 80)

    pca = PCAScratch(n_components=2)
    X_projected = pca.fit_transform(X)

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    total_var = (pc1_var + pc2_var)

    print(f"Original Feature Shape:  {X.shape}")
    print(f"Projected Feature Shape: {X_projected.shape}")
    print(f"PC1 Explained Variance:  {pc1_var:.2f}%")
    print(f"PC2 Explained Variance:  {pc2_var:.2f}%")
    print(f"Total Preserved Variance: {total_var:.2f}% (Over 80-90% retained in just 2 axes!)\n")

    # 4. Part 3: Clustering in PCA Subspace
    print("-" * 80)
    print("PART 3: K-MEANS ON COMPRESSED 2D PCA REPRESENTATION")
    print("-" * 80)

    optimal_k = 3
    kmeans_pca = KMeansScratch(n_clusters=optimal_k, max_iters=200, seed=42)
    kmeans_pca.fit(X_projected)

    # Check cluster distribution
    unique_labels, counts = np.unique(kmeans_pca.labels_, return_counts=True)
    for lbl, count in zip(unique_labels, counts):
        print(f"Cluster {lbl}: {count} samples assigned")
    print(f"Final 2D Inertia: {kmeans_pca.inertia_:.2f}")

    # Generate Visual Plots
    try:
        import matplotlib.pyplot as plt

        # Plot 1: Elbow Curve
        plt.figure(figsize=(8, 4.5))
        plt.plot(k_values, inertias, "bo-", linewidth=2, markersize=7)
        plt.title("K-Means Elbow Method: Finding Optimal Clusters", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Clusters (K)", fontsize=11)
        plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.annotate(
            "Optimal Elbow (K=3)",
            xy=(3, inertias[2]),
            xytext=(4, inertias[2] + (inertias[0] - inertias[-1]) * 0.2),
            arrowprops=dict(facecolor="crimson", shrink=0.08, width=2, headwidth=8),
            fontweight="bold",
            color="crimson"
        )
        plt.tight_layout()
        plt.savefig("/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/01_ml_fundamentals/04_kmeans_and_pca/kmeans_elbow_curve.png", dpi=150)
        print("\n[SUCCESS] Saved Elbow Curve to 'kmeans_elbow_curve.png'")

        # Plot 2: 2D PCA Clusters
        plt.figure(figsize=(9, 6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for k in range(optimal_k):
            cluster_data = X_projected[kmeans_pca.labels_ == k]
            plt.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                c=colors[k],
                label=f"Cluster {k}",
                alpha=0.7,
                edgecolors="k",
                s=40
            )

        # Plot Centroids
        plt.scatter(
            kmeans_pca.centroids[:, 0],
            kmeans_pca.centroids[:, 1],
            c="red",
            marker="X",
            s=200,
            linewidths=2,
            edgecolors="black",
            label="Centroids"
        )

        plt.title(f"PCA 2D Projection with K-Means (Total Variance Retained: {total_var:.1f}%)", fontsize=12, fontweight="bold")
        plt.xlabel(f"Principal Component 1 ({pc1_var:.1f}% Variance)", fontsize=11)
        plt.ylabel(f"Principal Component 2 ({pc2_var:.1f}% Variance)", fontsize=11)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/01_ml_fundamentals/04_kmeans_and_pca/pca_clusters_visualization.png", dpi=150)
        print("[SUCCESS] Saved PCA Cluster Plot to 'pca_clusters_visualization.png'")

    except ImportError:
        print("\n[INFO] matplotlib not installed; skipping graphic plot generation.")

if __name__ == "__main__":
    main()
