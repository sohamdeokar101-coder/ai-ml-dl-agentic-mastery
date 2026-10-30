import numpy as np
from linear_regression import LinearRegressionScratch
from logistic_regression import LogisticRegressionScratch

def run_linear_regression_demo():
    print("=" * 60)
    print("PART 1: LINEAR REGRESSION FROM SCRATCH")
    print("=" * 60)
    
    np.random.seed(42)
    # Generate 200 samples with 2 features
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    
    # Ground truth formula: y = 3.5 * X0 - 1.8 * X1 + 4.2 + noise
    true_weights = np.array([[3.5], [-1.8]])
    true_bias = 4.2
    noise = np.random.randn(n_samples, 1) * 0.5
    y = np.dot(X, true_weights) + true_bias + noise

    # Split train/test (80/20)
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = LinearRegressionScratch(learning_rate=0.05, n_iterations=600)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2 = model.r2_score(y_test, y_pred_test)

    print(f"True Weights: {true_weights.flatten()} | True Bias: {true_bias}")
    print(f"Learned Weights: {model.weights.flatten().round(4)} | Learned Bias: {round(float(model.bias), 4)}")
    print(f"Initial Loss: {model.loss_history[0]:.4f} -> Final Loss: {model.loss_history[-1]:.4f}")
    print(f"Test R^2 Score: {r2 * 100:.2f}%\n")


def run_logistic_regression_demo():
    print("=" * 60)
    print("PART 2: LOGISTIC REGRESSION FROM SCRATCH")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 300
    
    # Generate 2 distinct clusters
    class_0 = np.random.randn(n_samples // 2, 2) + np.array([-2.0, -2.0])
    class_1 = np.random.randn(n_samples // 2, 2) + np.array([2.0, 2.0])
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # Split train/test (80/20)
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    clf = LogisticRegressionScratch(learning_rate=0.1, n_iterations=800)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_test = clf.predict(X_test)
    acc = clf.accuracy_score(y_test, y_pred_test)

    print(f"Learned Weights: {clf.weights.flatten().round(4)}")
    print(f"Learned Bias: {round(float(clf.bias), 4)}")
    print(f"Initial BCE Loss: {clf.loss_history[0]:.4f} -> Final BCE Loss: {clf.loss_history[-1]:.4f}")
    print(f"Test Accuracy: {acc * 100:.2f}%\n")

    # Sample sample predictions
    sample_probs = clf.predict_proba(X_test[:5]).flatten()
    print("First 5 Test Predictions:")
    for idx, (p, pred, actual) in enumerate(zip(sample_probs, y_pred_test[:5].flatten(), y_test[:5])):
        print(f"  Sample {idx+1}: P(y=1)={p:.4f} => Pred={pred} (Actual={int(actual)})")

if __name__ == "__main__":
    run_linear_regression_demo()
    run_logistic_regression_demo()
