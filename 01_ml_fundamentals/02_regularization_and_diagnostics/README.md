# Project 02: Regularization, Polynomial Features & Overfitting Diagnostics

A from-scratch NumPy implementation of **Polynomial Feature Expansion**, **$Z$-Score Standardization (without data leakage)**, **L2 (Ridge)** and **L1 (Lasso) Regularization**, and **Bias-Variance Diagnostics**.

---

## 🎯 Concepts Mastered

1. **Polynomial Feature Expansion**:
   - Transforming non-linear data into higher dimensional feature spaces $[x^1, x^2, \dots, x^d]$ using clean vectorization (`np.hstack`).
2. **Preventing Data Leakage with $Z$-Score Standardization**:
   - Why high polynomial degrees cause gradient divergence without scaling.
   - Strictly fitting mean $\mu_{\text{train}}$ and std $\sigma_{\text{train}}$ on training data and reusing them on validation data (`axis=0` with $\epsilon$-guardrail).
3. **L2 (Ridge) Regularization (Weight Decay)**:
   - Penalty: $J = \text{MSE} + \frac{\lambda}{2m} \sum w_j^2$
   - Gradient: $\frac{\partial J}{\partial w} = \text{MSE}_{\text{grad}} + \frac{\lambda}{m} w$
   - Proportional weight decay smoothing out steep fluctuations.
4. **L1 (Lasso) Regularization (Feature Selection & Sparsity)**:
   - Penalty: $J = \text{MSE} + \frac{\lambda}{m} \sum |w_j|$
   - Gradient: $\frac{\partial J}{\partial w} = \text{MSE}_{\text{grad}} + \frac{\lambda}{m} \text{sign}(w)$
   - Constant push towards 0 that drives non-informative polynomial weights to **exactly zero**.
5. **Bias-Variance Diagnostics**:
   - Detecting **Underfitting (High Bias)**: High Train MSE, High Val MSE.
   - Detecting **Overfitting (High Variance)**: Low Train MSE, Exploding Val MSE.
   - Achieving the **Optimal Frontier**: Low Train MSE, Low Val MSE.

---

## 📂 Project Structure

```text
02_regularization_and_diagnostics/
├── README.md                  # In-depth theory, math & benchmark results
├── features.py                # polynomial_features() & StandardScalerScratch
├── regularized_regression.py  # RegularizedRegressionScratch (L1, L2, None)
├── diagnostics.py             # train_val_split() & evaluate_polynomial_pipeline()
├── main.py                    # Complete runnable experiment & weight sparsity comparison
└── overfitting_diagnostics.png # Visual comparison plot of all fitted curves
```

---

## 🚀 Running the Benchmark

```bash
cd "/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/01_ml_fundamentals/02_regularization_and_diagnostics"
python3 main.py
```
