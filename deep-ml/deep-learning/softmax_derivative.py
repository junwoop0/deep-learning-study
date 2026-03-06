# Problem: Derivative of Softmax
# URL: https://www.deep-ml.com/problems/219

import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	x = np.array(x, dtype = float)
	x = np.exp(x) / np.sum(np.exp(x))
	n = x.size
	arr = np.zeros((n,n), dtype = float)
	for i in range(n):
		for j in range (n):
			if (i == j):
				arr[i][j] = x[i] * (1 - x[i])
			if (i != j):
				arr[i][j] = -1 * x[i] * x[j]
	return arr

'''
Notes
- np.exp() is used to compute the exponential of each element in the input array
- np.sum() computes the sum of array elements over a specified axis
    - ex) np.sum(arr, axis=0) computes the sum along the columns (i.e., for each column)
- To understand the principles of Jacobian matrix of softmax, you should write it down on paper and derive it by hand
- np.diag() creates a diagonal matrix from a 1D array, and np.outer() computes the outer product of two vectors
'''

'''
ChatGPT Solution - 1

import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.

    Args:
        x: Input vector of real numbers

    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i) / d(x_j)
    """
    x = np.array(x, dtype=float)

    # Apply a numerically stable softmax
    exp_x = np.exp(x - np.max(x))
    s = exp_x / np.sum(exp_x)

    # Jacobian of softmax: diag(s) - s s^T
    jacobian = np.diag(s) - np.outer(s, s)

    return jacobian.tolist()

'''