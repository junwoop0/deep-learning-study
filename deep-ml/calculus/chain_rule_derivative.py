# Problem: Chain Rule for Composite Functions
# URL: https://www.deep-ml.com/problems/214

import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	forward = {
		"sin": lambda z: np.sin(z),
		"square": lambda z: z**2,
		"exp": lambda z: np.exp(z),
		"log": lambda z: np.log(z),
	}
	backward = {
		"sin": lambda z: np.cos(z),
		"square": lambda z: 2*z,
		"exp": lambda z: np.exp(z),
		"log": lambda z: 1 / z,
	}
	
	# make forward list
	forward_list = [forward[functions[-1]](x)]
	for i in range(1, len(functions)):
		forward_list.append(forward[functions[len(functions) - i - 1]](forward_list[i-1]))
    # make backward list
	backward_list = [backward[functions[-1]](x)]
	for i in range(1, len(functions)):
		backward_list.append(backward[functions[len(functions) - i - 1]](forward_list[i-1]))
	ans = 1.0
	for i in range(len(backward_list)):
		ans *= backward_list[i]

	return ans

'''
Notes
- If you want to define functions, you can use table
- How to define functions in table:
    - func_dict = {
        'square': lambda x: x**2,
        'sin': lambda x: np.sin(x),
        'exp': lambda x: np.exp(x),
        'log': lambda x: np.log(x)
    }
    - To use it, you can call func_dict['square'](x) to compute x²
- When applying the chain rule or multiplying/differentiating polynomials, always think of making lists.
- To make reversed list, use reversed() function or use range() function with negative step
- To use each elements in both lists, use zip() function
    - for f, b in zip(reversed(functions), z_inputs):
'''

'''
ChatGPT Solution - 1
def compute_chain_rule_gradient_clean(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    func_dict = {
        'square': lambda z: z**2,
        'sin': lambda z: np.sin(z),
        'exp': lambda z: np.exp(z),
        'log': lambda z: np.log(z)
    }
    # make forward list
    forward_list = [func_dict[functions[-1]](x)]
    for i in range(1, len(functions)):
        forward_list.append(func_dict[functions[len(functions) - i - 1]](forward_list[i-1]))
    # make backward list
    backward_list = []
    for i in range(len(functions)):
        if functions[len(functions) - 1 - i] == 'square':
            backward_list.append(2 * forward_list[i])
        elif functions[len(functions) - 1 - i] == 'sin':
            backward_list.append(np.cos(forward_list[i]))
        elif functions[len(functions) - 1 - i] == 'exp':
            backward_list.append(np.exp(forward_list[i]))
        elif functions[len(functions) - 1 - i] == 'log':
            backward_list.append(1 / forward_list[i])
    ans = 1.0
    for i in range(len(backward_list)):
        ans *= backward_list[i]

    return ans
'''