# Problem: Linear Regression Using Gradient Descent
# URL: https://www.deep-ml.com/problems/15

import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    for i in range(iterations):
        mul_mat = X @ theta
        loss = mul_mat - y
        # MSE = np.mean(np.square(loss)) / 2
        grad = (X.T @ loss) / m
        theta = theta - (grad * alpha)
    
    return theta.flatten()

'''
Notes
- you have to study the concept of linear algebra to solve this problem.
'''