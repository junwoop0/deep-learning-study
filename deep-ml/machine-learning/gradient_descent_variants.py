# Problem: Implement Gradient Descent Variants with MSE Loss
# URL: https://www.deep-ml.com/problems/47

import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape
    if method == 'batch':
        for i in range(n_epochs):
            mul_mat = X @ weights
            loss = mul_mat - y
            grad = 2 * (X.T @ loss) / m
            weights -= grad * learning_rate
    if method == 'stochastic':
        for i in range(n_epochs):
            for j in range(m):
                mul_mat = X[j] @ weights
                loss = mul_mat - y[j]
                grad = 2 * (X[j].T * loss)
                weights -= grad * learning_rate
    if method == 'mini_batch':
        for i in range(n_epochs):
            iterate = m // batch_size
            for j in range (iterate):
                x_min = j * batch_size
                x_max = (j+1) * batch_size
                mul_mat = X[x_min:x_max] @ weights
                loss = mul_mat - y[x_min:x_max]
                grad = (2 * X[x_min:x_max].T @ loss) / batch_size
                weights -= grad * learning_rate

    return weights


'''
Notes
- Slicing doesn't include the end index, so to get the first 10 elements, you can use x[:10].
- When loss is scalar, you can't use @ operator, you can just use * operator for multiplication.
'''
