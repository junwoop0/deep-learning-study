# Problem: Quotient Rule for Derivatives
# URL: https://www.deep-ml.com/problems/312

import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g_deri = []
    for i in range(len(g_coeffs)-1, 0, -1):
        g_deri.append((i * g_coeffs[len(g_coeffs)-1 - i]))
    if g_deri == []:
        g_deri = [0]
    h_deri = []
    for i in range(len(h_coeffs)-1, 0, -1):
        h_deri.append((i * h_coeffs[len(h_coeffs)-1 - i]))
    if h_deri == []:
        h_deri = [0]
    first_num = np.convolve(g_deri, h_coeffs)
    second_num = np.convolve(h_deri, g_coeffs)
    first_num_val = 0
    second_num_val = 0
    for i in range(len(first_num)):
        first_num_val += first_num[i] * x ** (len(first_num)-1-i)
    for i in range(len(second_num)):
        second_num_val += second_num[i] * x ** (len(second_num)-1-i)
    denom_val = (np.polyval(h_coeffs, x))**2
    ans = (first_num_val - second_num_val) / denom_val 
    return ans

'''
Notes
- Using for loop, to make the integer in descending order, use range(start, stop, step)
    - if the step in negative, the integer will stop before the stop value
- If you want to multiply two polynomials using numpy, use np.convolve() function
    - when you should compute value of it, use np.polyval() function
- How np.convolve() works:
    - If you have two polynomials, g(x) = 2x^2 + 3 and h(x) = x + 1, the coefficients are:
        - g_coeffs = [2, 0, 3]
        - h_coeffs = [1, 1]
    - To compute the product g(x)*h(x), you can use np.convolve(g_coeffs, h_coeffs)
    - The result will be the coefficients of the resulting polynomial in descending order
- How np.polyval() works:
    - If you have a polynomial with coefficients in descending order, for example, p(x) = 2x^2 + 0x + 3, the coefficients are p_coeffs = [2, 0, 3]
    - To evaluate p(x) at a specific value of x, you can use np.polyval(p_coeffs, x)
- Never forget to check if list is empty
'''

'''
Gemini Solution - 1

def quotient_rule_derivative_clean(g_coeffs: list, h_coeffs: list, x: float) -> float:
    # 1. Compute derivative coefficients (using np.polyder)
    g_deri = np.polyder(g_coeffs)
    h_deri = np.polyder(h_coeffs)
    
    # 2. Calculate the numerator of the quotient rule: g'(x)h(x) - g(x)h'(x)
    term1 = np.polyval(g_deri, x) * np.polyval(h_coeffs, x)
    term2 = np.polyval(g_coeffs, x) * np.polyval(h_deri, x)
    
    # 3. Calculate the denominator: h(x)^2
    denom_val = np.polyval(h_coeffs, x) ** 2
    
    # 4. Final result
    return (term1 - term2) / denom_val
    
'''