# Problem: Derivative of Cross-Entropy Loss w.r.t. Logits
# URL: https://www.deep-ml.com/problems/220

import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	x = np.array(logits)
	x = np.exp(x - np.max(x))
	x = x / np.sum(x)
	tar_list = np.zeros(np.size(x), dtype = float)
	tar_list[target] = 1
	ans = x - tar_list
	
	return ans

'''
Notes
- numpy array and python list are little different, so you should need to learn how to use numpy array
'''

'''
ChatGPT Solution - 1
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    """
    Compute the derivative of cross-entropy loss with respect to logits.

    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class (0-indexed)

    Returns:
        Gradient vector where gradient[i] = dL / d(logits[i])
    """
    logits = np.array(logits, dtype=float)

    # Apply a numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Gradient of cross-entropy loss with softmax: p - y
    probs[target] -= 1.0

    return probs.tolist()
'''
