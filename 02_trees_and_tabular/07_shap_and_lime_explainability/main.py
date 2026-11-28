import os
import sys
import numpy as np

sys.path.append(os.path.abspath("../06_random_forest_and_xgboost"))
from gradient_boosting import GradientBoostingRegressorScratch
from explainers import (
    ShapleyExplainerScratch,
    LimeExplainerScratch,
    permutation_feature_importance,
)


def generate_credit_risk_data(n_samples: int = 400, seed: int = 42):
    """
    Generates a synthetic tabular Credit Default Risk dataset with 4 features:
        0. Debt_to_Income (0.1 to 0.9)      -> Strong positive risk driver
        1. Credit_History_Years (0 to 20)   -> Negative risk driver (lowers risk)
        2. Missed_Payments (0 to 5)         -> Non-linear threshold risk driver
        3. Random_Noise_ID (0 to 100)       -> Pure uninformative noise
    """
    rng = np.random.default_rng(seed)

    debt_to_income = rng.uniform(0.1, 0.9, size=n_samples)
    credit_history = rng.uniform(0.0, 20.0, size=n_samples)
    missed_payments = rng.integers(0, 6, size=n_samples).astype(float)
    random_noise_id = rng.uniform(0.0, 100.0, size=n_samples)

    X = np.column_stack([debt_to_income, credit_history, missed_payments, random_noise_id])

    # Non-linear ground truth risk score in roughly [0, 1]
    raw_risk = (
        0.55 * debt_to_income
        - 0.015 * credit_history
        + 0.10 * (missed_payments ** 1.3)
        + 0.12 * (debt_to_income * (missed_payments >= 2))
        + rng.normal(0.0, 0.03, size=n_samples)
    )
    y = np.clip(raw_risk, 0.02, 0.98)

    feature_names = [
        "Debt_to_Income",
        "Credit_History_Years",
        "Missed_Payments",
        "Random_Noise_ID",
    ]
    return X, y, feature_names


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    val_size = int(len(X) * val_ratio)
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def main():
    print("=" * 80)
    print("PROJECT 07: MODEL EXPLAINABILITY & INTERPRETABILITY (SHAP & LIME)")
    print("=" * 80)

    X, y, feature_names = generate_credit_risk_data(n_samples=400, seed=42)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.25, seed=42)

    # Train Gradient Boosted Trees black-box model
    model = GradientBoostingRegressorScratch(n_estimators=60, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_r2 = GradientBoostingRegressorScratch.r2_score(y_val, y_val_pred)
    print(f"Trained Gradient Boosting Black-Box Model | Validation R^2 = {val_r2 * 100:.2f}%\n")

    # --------------------------------------------------------------------------
    # PART 1: GLOBAL PERMUTATION FEATURE IMPORTANCE
    # --------------------------------------------------------------------------
    print("-" * 80)
    print("PART 1: GLOBAL PERMUTATION FEATURE IMPORTANCE (VALIDATION SET)")
    print("-" * 80)

    mean_imp, std_imp = permutation_feature_importance(
        model.predict,
        X_val,
        y_val,
        metric_fn=GradientBoostingRegressorScratch.r2_score,
        n_repeats=15,
        seed=42,
    )

    print(f"{'Feature':<24} | {'Importance (R^2 Drop)':<22} | {'Std Dev':<10}")
    print("-" * 64)
    for name, m_val, s_val in sorted(zip(feature_names, mean_imp, std_imp), key=lambda t: -t[1]):
        print(f"{name:<24} | {m_val:<22.4f} | {s_val:<10.4f}")

    # --------------------------------------------------------------------------
    # PART 2: SHAPLEY VALUES (EXACT VS. MONTE CARLO) & LOCAL ACCURACY AXIOM
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 2: SHAP LOCAL EXPLANATION (HIGH-RISK CUSTOMER AUDIT)")
    print("-" * 80)

    # Pick the highest-risk customer in validation set
    target_idx = int(np.argmax(y_val_pred))
    x_target = X_val[target_idx]
    f_x = float(model.predict(x_target.reshape(1, -1))[0])

    # Use a 60-sample background summary for fast exact coalition expectation
    background_X = X_train[:60]
    shap_explainer = ShapleyExplainerScratch(model.predict, background_X=background_X)

    phi_exact = shap_explainer.explain_exact(x_target)
    phi_mc = shap_explainer.explain_monte_carlo(x_target, n_samples=250, seed=42)

    phi_0 = shap_explainer.expected_value_
    reconstructed = phi_0 + float(np.sum(phi_exact))

    print(f"Target Customer Features: {dict(zip(feature_names, np.round(x_target, 3)))}")
    print(f"Baseline Expected Risk  E[f(X)] (phi_0) : {phi_0:.4f}")
    print(f"Model Actual Prediction f(x*)           : {f_x:.4f}")
    print(f"SHAP Sum (phi_0 + sum(phi_i))           : {reconstructed:.4f} (Exact Local Accuracy Diff = {abs(f_x - reconstructed):.2e})\n")

    print(f"{'Feature':<24} | {'Value':<8} | {'Exact SHAP':<12} | {'Monte Carlo SHAP':<16}")
    print("-" * 68)
    for name, val, pe, pmc in zip(feature_names, x_target, phi_exact, phi_mc):
        print(f"{name:<24} | {val:<8.3f} | {pe:+12.4f} | {pmc:+16.4f}")

    # --------------------------------------------------------------------------
    # PART 3: LIME LOCAL SURROGATE EXPLANATION
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 3: LIME LOCAL LINEAR SURROGATE (STANDARDIZED WEIGHTS)")
    print("-" * 80)

    lime_explainer = LimeExplainerScratch(model.predict, training_X=X_train, kernel_width=1.5, reg_lambda=1e-2)
    lime_intercept, lime_weights = lime_explainer.explain(x_target, n_perturbations=1000, seed=42)

    print(f"LIME Local Surrogate Intercept g(x*): {lime_intercept:.4f} (vs Black-Box f(x*): {f_x:.4f})")
    print(f"{'Feature':<24} | {'LIME Local Weight (per 1 std)':<30}")
    print("-" * 58)
    for name, w in zip(feature_names, lime_weights):
        direction = "Increases Risk (+)" if w > 0.01 else ("Decreases Risk (-)" if w < -0.01 else "Neutral (~0)")
        print(f"{name:<24} | {w:+12.4f}  ({direction})")

    # --------------------------------------------------------------------------
    # PART 4: VISUALIZATION DASHBOARD
    # --------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Left Panel: Global Permutation Feature Importance
        order = np.argsort(mean_imp)
        ax1.barh(
            [feature_names[i] for i in order],
            mean_imp[order],
            xerr=std_imp[order],
            color="#2B6CB0",
            alpha=0.85,
            capsize=4,
        )
        ax1.set_title("Global Permutation Feature Importance (R² Drop)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Mean R² Drop When Feature Shuffled", fontsize=10)
        ax1.grid(True, axis="x", alpha=0.3)

        # Right Panel: Local SHAP Attribution Waterfall Bars
        shap_order = np.argsort(np.abs(phi_exact))
        bar_colors = ["#C53030" if phi_exact[i] > 0 else "#2F855A" for i in shap_order]
        ax2.barh(
            [f"{feature_names[i]} = {x_target[i]:.2f}" for i in shap_order],
            phi_exact[shap_order],
            color=bar_colors,
            alpha=0.88,
        )
        ax2.axvline(0.0, color="black", linewidth=1)
        ax2.set_title(f"Local SHAP Attribution (Base {phi_0:.2f} → Pred {f_x:.2f})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("SHAP Value φ_i (Contribution to Risk Score)", fontsize=10)
        ax2.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        plot_path = "/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/02_trees_and_tabular/07_shap_and_lime_explainability/explainability_dashboard.png"
        plt.savefig(plot_path, dpi=150)
        print("\n[SUCCESS] Saved explainability dashboard plot to 'explainability_dashboard.png'")
    except ImportError:
        print("\n[INFO] matplotlib not installed; skipping plot generation.")


if __name__ == "__main__":
    main()
