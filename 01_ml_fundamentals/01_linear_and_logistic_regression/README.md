# Project 01: Linear & Logistic Regression from Scratch (NumPy)

A pure NumPy implementation of **Linear Regression** and **Binary Logistic Regression** without external machine learning frameworks (no Scikit-Learn, PyTorch, or TensorFlow).

---

## 🎯 Objectives & Concepts Mastered

1. **Analytical Gradients & Parameter Updates**:
   - Manually formulating partial derivatives with respect to weights ($w$) and bias ($b$).
   - Implementing batch gradient descent parameter updates:
     $$w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \quad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
2. **Vectorization**:
   - Eliminating manual Python `for` loops across training samples using matrix operations (`X @ w`, `X.T @ error`).
3. **Loss Functions**:
   - **Mean Squared Error (MSE)** for regression.
   - **Binary Cross-Entropy (BCE)** for classification.
4. **Numerical Stability**:
   - Preventing sigmoid exponent overflow ($e^{-z}$) via `np.clip(z, -500, 500)`.
   - Avoiding undefined $\log(0) = -\infty$ in BCE loss using small $\epsilon = 10^{-15}$ clipping.

---

## 📐 Mathematical Formulation

### 1. Linear Regression
- **Hypothesis**:
  $$\hat{y} = Xw + b$$
- **Cost Function (Mean Squared Error)**:
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \|Xw + b - y\|_2^2$$
- **Gradients**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

---

### 2. Logistic Regression (Binary Classification)
- **Logit**:
  $$z = Xw + b$$
- **Sigmoid Activation**:
  $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Cost Function (Binary Cross-Entropy)**:
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
- **Gradients**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$
  *(Notice the gradient form is identical to linear regression due to the derivative properties of the sigmoid activation combined with log loss!)*

---

## 📂 Project Structure

```text
01_linear_and_logistic_regression/
├── README.md               # Detailed math & documentation (this file)
├── linear_regression.py    # LinearRegressionScratch class & R^2 metric
├── logistic_regression.py  # LogisticRegressionScratch class & Accuracy metric
└── main.py                 # End-to-end data generation, training & evaluation
```

---

## 🚀 Running the Project

Run the verification pipeline directly from your terminal:

```bash
cd "/Volumes/External SSD/Projects/AI Engineer/ai-ml-dl-agentic-mastery/01_ml_fundamentals/01_linear_and_logistic_regression"
python3 main.py
```

### Expected Output

```text
============================================================
PART 1: LINEAR REGRESSION FROM SCRATCH
============================================================
True Weights: [ 3.5 -1.8] | True Bias: 4.2
Learned Weights: [ 3.5486 -1.7808] | Learned Bias: 4.1428
Initial Loss: 15.7986 -> Final Loss: 0.1207
Test R^2 Score: 98.25%

============================================================
PART 2: LOGISTIC REGRESSION FROM SCRATCH
============================================================
Learned Weights: [2.1077 1.7338]
Learned Bias: 0.0509
Initial BCE Loss: 0.6931 -> Final BCE Loss: 0.0112
Test Accuracy: 100.00%

First 5 Test Predictions:
  Sample 1: P(y=1)=0.0009 => Pred=0 (Actual=0)
  Sample 2: P(y=1)=1.0000 => Pred=1 (Actual=1)
  Sample 3: P(y=1)=1.0000 => Pred=1 (Actual=1)
  Sample 4: P(y=1)=0.0000 => Pred=0 (Actual=0)
  Sample 5: P(y=1)=0.0001 => Pred=0 (Actual=0)
```
