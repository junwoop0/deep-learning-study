# Problem: Partial Derivatives of Multivariable Functions
# URL: https://www.deep-ml.com/problems/215

import numpy as np

import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	if len(point) == 2:
		x = point[0]
		y = point[1]
	if len(point) == 3:
		x = point[0]
		y = point[1]
		z = point[2]

	if func_name == 'poly2d':
		par_x = [2 * y, y **2]
		par_y = [2 * x, x **2]
		grad_x = np.polyval(par_x, x)
		grad_y = np.polyval(par_y, y)
		grad = (grad_x, grad_y)

	if func_name == 'exp_sum':
		grad_x = np.exp(x + y)
		grad_y = np.exp(x + y)
		grad = (grad_x, grad_y)
	
	if func_name == 'product_sin':
		grad_x = np.sin(y)
		grad_y = x * np.cos(y)
		grad = (grad_x, grad_y)

	if func_name == 'poly3d':
		par_x = [2 * y, 0]
		par_y = [x**2 + z**2]
		par_z = [2*y, 0]
		grad_x = np.polyval(par_x, x)
		grad_y = np.polyval(par_y, y)
		grad_z = np.polyval(par_z, z)
		grad = (grad_x, grad_y, grad_z)

	if func_name == 'squared_error':
		grad_x = 2 * (x - y)
		grad_y = 2 * (x - y) * -1
		grad = (grad_x, grad_y)

	return grad

'''
Notes
- If you can't find a way to solve it cleanly, just write the code naively
- To unpack a tuple, you can use tuple unpacking
    - x, y = (tuple name)
        - example) x, y = point    
'''

'''
Gemini Solution - 1

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions.
    """
    
    if func_name == 'poly2d':
        x, y = point  # 1. Utilize tuple unpacking
        # 2. Write intuitive formulas directly (instead of np.polyval)
        return (2 * x * y + y**2, x**2 + 2 * x * y)
        
    elif func_name == 'exp_sum':
        x, y = point
        val = np.exp(x + y)  # Prevent redundant calculations
        return (val, val)
        
    elif func_name == 'product_sin':
        x, y = point
        return (np.sin(y), x * np.cos(y))
        
    elif func_name == 'poly3d':
        x, y, z = point
        return (2 * x * y, x**2 + z**2, 2 * y * z)
        
    elif func_name == 'squared_error':
        x, y = point
        diff = 2 * (x - y)
        return (diff, -diff)
        
    # Defensive programming: Handle unexpected function names
    raise ValueError(f"Unknown function: {func_name}")

'''