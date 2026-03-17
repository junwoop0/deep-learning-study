# Problem: Numerical Gradient Checking
# URL: https://www.deep-ml.com/problems/313

import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    size = np.size(x)
    num_grad = np.zeros(size)
    for i in range(size):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        num_grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    analytical_grad_size = np.linalg.norm(analytical_grad)
    num_grad_size = np.linalg.norm(num_grad)
    rel_err_size = np.linalg.norm(analytical_grad - num_grad)
    if (analytical_grad_size == 0) and (num_grad_size == 0):
        return (num_grad, 0)
    rel_err = rel_err_size / (analytical_grad_size + num_grad_size)

    return (num_grad, rel_err)

'''
Notes
- centered finite difference method: f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)
- To see the size of a vector, use n = np.linalg.norm(v)
    - This is the same as np.sqrt(np.sum(v**2))
'''