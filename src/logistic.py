import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid function."""
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def hypothesis(X, theta):
    """Return predicted fault probabilities for design matrix X."""
    return sigmoid(X @ theta)


def cost(X, y, theta):
    """Mean binary cross-entropy, with probabilities clipped as required."""
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)))


def fit_logistic(X, y, alpha, n_iters=2000):
    """Batch gradient descent from zero theta; returns theta and cost history."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.zeros(X.shape[1], dtype=float)
    costs = [cost(X, y, theta)]
    m = len(y)
    for _ in range(n_iters):
        h = hypothesis(X, theta)
        gradient = (X.T @ (h - y)) / m
        theta = theta - alpha * gradient
        costs.append(cost(X, y, theta))
    return theta, np.asarray(costs)


def predict(X, theta):
    """Predict 1 when the probability is at least 0.5; otherwise 0."""
    return (hypothesis(X, theta) >= 0.5).astype(int)
