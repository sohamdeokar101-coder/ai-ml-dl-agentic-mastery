import numpy as np
from features import polynomial_features, StandardScalerScratch
from regularized_regression import RegularizedRegressionScratch

def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.25, seed: int = 42):
    """Splits X and y into training and validation sets deterministically."""
    np.random.seed(seed)
    m = len(X)
    indices = np.random.permutation(m)
    val_size = int(m * val_ratio)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def evaluate_polynomial_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    degree: int,
    penalty: str = "none",
    lambda_reg: float = 0.0,
    lr: float = 0.05,
    n_iterations: int = 1500
):
    """
    End-to-end pipeline:
    1. Polynomial expansion on train and val
    2. Fit scaler ONLY on train, transform both train and val
    3. Fit RegularizedRegressionScratch on scaled train data
    4. Compute Train MSE & Val MSE
    """
    # 1. Feature Expansion
    X_train_poly = polynomial_features(X_train, degree)
    X_val_poly = polynomial_features(X_val, degree)

    # 2. Standardization without leakage
    scaler = StandardScalerScratch()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_val_scaled = scaler.transform(X_val_poly)

    # 3. Model training
    model = RegularizedRegressionScratch(
        penalty=penalty,
        lambda_reg=lambda_reg,
        learning_rate=lr,
        n_iterations=n_iterations
    )
    model.fit(X_train_scaled, y_train)

    # 4. Evaluation
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_mse = RegularizedRegressionScratch.mse(y_train, y_train_pred)
    val_mse = RegularizedRegressionScratch.mse(y_val, y_val_pred)
    val_r2 = RegularizedRegressionScratch.r2_score(y_val, y_val_pred)

    return {
        "degree": degree,
        "penalty": penalty,
        "lambda_reg": lambda_reg,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "val_r2": val_r2,
        "weights": model.weights.flatten(),
        "bias": float(model.bias)
    }
