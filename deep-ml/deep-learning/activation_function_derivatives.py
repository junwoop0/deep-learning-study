# Problem: Derivatives of Activation Functions
# URL: https://www.deep-ml.com/problems/217

import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	sig = (1 / (1 + np.exp(-x))).item()
	tanh = (np.tanh(x)).item()
	ans = {
		"sigmoid" : sig * (1 - sig),
		"tanh" : 1 - (tanh)**2,
		"relu" : (np.where(x > 0, 1.0, 0.0)).item()
	}
	
	return ans

'''
Notes
- To implement RELU using numpy, use np.maximum(0, x)
- To make condition using numpy, use np.where(condition, value_if_true, value_if_false) or 
value_if_true if condition else value_if_false
- To make numpy vectors to float, use .item() method
'''