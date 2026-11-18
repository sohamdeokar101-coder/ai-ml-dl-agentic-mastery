import numpy as np
from diagnostics import train_val_split, evaluate_polynomial_pipeline

def generate_synthetic_data(n_samples: int = 80, noise_std: float = 0.15, seed: int = 42):
    """Generates non-linear data following a cosine wave: y = cos(1.5 * pi * x) + noise"""
    np.random.seed(seed)
    X = np.sort(np.random.uniform(0, 1, size=(n_samples, 1)), axis=0)
    y = np.cos(1.5 * np.pi * X) + np.random.randn(n_samples, 1) * noise_std
    return X, y

def main():
    print("=" * 80)
    print("PROJECT 02: REGULARIZATION & OVERFITTING DIAGNOSTICS BENCHMARK")
    print("=" * 80)

    # 1. Generate non-linear data
    X, y = generate_synthetic_data(n_samples=60, noise_std=0.2, seed=42)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.3, seed=42)

    print(f"Dataset generated: {len(X_train)} Train samples, {len(X_val)} Validation samples\n")

    # 2. Define configurations to benchmark
    experiments = [
        {"name": "1. Linear (Degree 1, No Reg)", "degree": 1, "penalty": "none", "lambda_reg": 0.0},
        {"name": "2. High Capacity (Degree 10, No Reg)", "degree": 10, "penalty": "none", "lambda_reg": 0.0},
        {"name": "3. Ridge L2 (Degree 10, lambda=0.5)", "degree": 10, "penalty": "l2", "lambda_reg": 0.5},
        {"name": "4. Lasso L1 (Degree 10, lambda=0.2)", "degree": 10, "penalty": "l1", "lambda_reg": 0.2},
    ]

    results = []
    print(f"{'Experiment':<35} | {'Train MSE':<10} | {'Val MSE':<10} | {'Val R^2 (%)':<12} | {'Diagnosis'}")
    print("-" * 88)

    for exp in experiments:
        res = evaluate_polynomial_pipeline(
            X_train, y_train, X_val, y_val,
            degree=exp["degree"],
            penalty=exp["penalty"],
            lambda_reg=exp["lambda_reg"],
            lr=0.1,
            n_iterations=2500
        )
        
        # Diagnostic heuristic
        if res["train_mse"] > 0.15 and res["val_mse"] > 0.15:
            diagnosis = "Underfitting (High Bias)"
        elif res["val_mse"] > 1.8 * res["train_mse"]:
            diagnosis = "Overfitting (High Variance)"
        else:
            diagnosis = "Good Generalization"

        print(f"{exp['name']:<35} | {res['train_mse']:<10.4f} | {res['val_mse']:<10.4f} | {res['val_r2']*100:<12.2f} | {diagnosis}")
        results.append((exp["name"], res))

    print("\n" + "=" * 80)
    print("WEIGHT SPARSITY & MAGNITUDE COMPARISON (Degree 10)")
    print("=" * 80)
    
    unreg_weights = results[1][1]["weights"]
    ridge_weights = results[2][1]["weights"]
    lasso_weights = results[3][1]["weights"]

    print(f"{'Feature':<10} | {'Unregularized w':<16} | {'Ridge (L2) w':<14} | {'Lasso (L1) w':<14}")
    print("-" * 65)
    for i in range(10):
        print(f"x^{i+1:<7} | {unreg_weights[i]:<16.4f} | {ridge_weights[i]:<14.4f} | {lasso_weights[i]:<14.4f}")

    # Count zero weights (sparsity) in Lasso vs Ridge
    lasso_zeros = np.sum(np.abs(lasso_weights) < 1e-3)
    ridge_zeros = np.sum(np.abs(ridge_weights) < 1e-3)
    print("-" * 65)
    print(f"Near-Zero Weights (< 0.001): Ridge = {ridge_zeros} / 10  vs  Lasso = {lasso_zeros} / 10 (Sparse Feature Selection!)")

    # Optional plot generation if matplotlib exists
    try:
        import matplotlib.pyplot as plt
        
        X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
        y_true = np.cos(1.5 * np.pi * X_plot)

        plt.figure(figsize=(12, 6))
        plt.scatter(X_train, y_train, color="blue", label="Train Data", alpha=0.7)
        plt.scatter(X_val, y_val, color="red", label="Validation Data", alpha=0.7)
        plt.plot(X_plot, y_true, "k--", label="True Function", alpha=0.5)

        for name, res in results:
            deg = res["degree"]
            pen = res["penalty"]
            lam = res["lambda_reg"]
            
            # Predict on smooth line
            from features import polynomial_features, StandardScalerScratch
            from regularized_regression import RegularizedRegressionScratch
            
            X_tr_poly = polynomial_features(X_train, deg)
            X_pl_poly = polynomial_features(X_plot, deg)
            
            scaler = StandardScalerScratch()
            X_tr_sc = scaler.fit_transform(X_tr_poly)
            X_pl_sc = scaler.transform(X_pl_poly)
            
            model = RegularizedRegressionScratch(penalty=pen, lambda_reg=lam, learning_rate=0.1, n_iterations=2500)
            model.fit(X_tr_sc, y_train)
            y_curve = model.predict(X_pl_sc)
            
            plt.plot(X_plot, y_curve, label=name)

        plt.ylim(-1.8, 1.8)
        plt.title("Bias-Variance Tradeoff: Polynomial Fits with Ridge & Lasso Regularization")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/01_ml_fundamentals/02_regularization_and_diagnostics/overfitting_diagnostics.png", dpi=150)
        print("\n[SUCCESS] Generated visual curve comparison saved to 'overfitting_diagnostics.png'")
    except ImportError:
        print("\n[INFO] matplotlib not installed; skipping graphic plot generation.")

if __name__ == "__main__":
    main()
