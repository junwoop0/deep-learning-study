# Problem: Gradient Direction and Magnitude
# URL: https://www.deep-ml.com/problems/308

import numpy as np

import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	mag = np.linalg.norm(gradient)
	grad_new = np.array(gradient)
	direct = grad_new / mag
	des_direct = direct * -1
	direct = direct.tolist()
	des_direct = des_direct.tolist()
	if mag == 0:
		direct = [0,0]
		des_direct = [0,0]
	ans = {
		"magnitude" : mag,
		"direction" : direct,
		"descent_direction" : des_direct
	}

	return ans

'''
Notes
- To calculate the magnitude or the distance of a vector, use numpy.linalg.norm() function
    - np.linalg.norm(x, ord=None, axis=None, keepdims=False)
        - x: input array
        - ord: order of the norm (default is 2 for L2 norm)
- To create a unit vector, divide the original vector by its magnitude
- Norm means the length or the size of the vector
- If you make the list in numpy array, you can divide the list by a scalar value without using a for loop
- If you want to change the type in the list using numpy, use .astype() method
    - np.array(gradient).astype(float)
- If you want to make numpy list to python list, use .tolist() method
    - gradient.tolist()
'''

'''
Gemini Solution - 1

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    """
    grad_arr = np.array(gradient)
    mag = np.linalg.norm(grad_arr)
    
    # check if magnitude is zero at the beginning to avoid unnecessary calculations
    if mag == 0:
        # make empty list which is applicable for any length of gradient
        zero_list = [0.0] * len(gradient) 
        return {
            "magnitude": 0.0,
            "direction": zero_list,
            "descent_direction": zero_list
        }
        
    direction = grad_arr / mag
    
    descent_direction = -direction 
    
    # Use early return pattern
    return {
        "magnitude": float(mag), 
        "direction": direction.tolist(),
        "descent_direction": descent_direction.tolist()
    }
'''