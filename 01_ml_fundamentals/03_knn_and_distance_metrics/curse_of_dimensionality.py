import numpy as np

def run_curse_of_dimensionality_experiment(
    dimensions: list = [2, 5, 10, 25, 50, 100, 200, 500],
    n_samples: int = 500,
    seed: int = 42
):
    """
    Demonstrates the Distance Concentration Effect in high dimensions:
    For a set of uniformly distributed random points in [0, 1]^D,
    measures min_distance, max_distance, and the contrast ratio:
        (max_dist - min_dist) / min_dist
    As D increases, the contrast ratio approaches 0 (all points become equidistant).
    """
    np.random.seed(seed)
    results = []

    for d in dimensions:
        # Generate n_samples in D-dimensional hypercube
        X = np.random.uniform(0, 1, size=(n_samples, d))
        
        # Pick point 0 as query point and measure distances to all other points
        query_pt = X[0:1] # shape (1, d)
        other_pts = X[1:] # shape (n_samples - 1, d)
        
        # Compute Euclidean distances
        dists = np.sqrt(np.sum((other_pts - query_pt) ** 2, axis=1))
        
        d_min = float(np.min(dists))
        d_max = float(np.max(dists))
        contrast_ratio = float((d_max - d_min) / (d_min + 1e-9))
        
        results.append({
            "dimension": d,
            "min_dist": d_min,
            "max_dist": d_max,
            "contrast_ratio": contrast_ratio
        })

    return results
