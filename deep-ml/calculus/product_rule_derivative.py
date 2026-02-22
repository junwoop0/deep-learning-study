# Problem: Product Rule for Derivatives
# URL: https://www.deep-ml.com/problems/309

import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    f_deri = []
    for i in range(len(f_coeffs)-1):
        f_deri.append(f_coeffs[i+1]*(i+1))
    g_deri = []
    for i in range(len(g_coeffs)-1):
        g_deri.append(g_coeffs[i+1]*(i+1))
    ans_list_1 = [0] * (len(f_deri)+len(g_coeffs)-1)
    for o1, i1 in enumerate(f_deri):
        for o2, i2 in enumerate (g_coeffs):
            ans_list_1[o1+o2] += i1 * i2
    ans_list_2 = [0] * (len(g_deri)+len(f_coeffs)-1)
    for o1, i1 in enumerate(g_deri):
        for o2, i2 in enumerate (f_coeffs):
            ans_list_2[o1+o2] += i1 * i2
    if len(ans_list_1) > len(ans_list_2):
        sub = len(ans_list_1) - len(ans_list_2)
        for i in range(sub):
            ans_list_2.append(0)
    if len(ans_list_2) > len(ans_list_1):
        sub = len(ans_list_2) - len(ans_list_1)
        for i in range(sub):
            ans_list_1.append(0)  
    ans_list = []
    for i in range(len(ans_list_1)):
        ans_list.append(round(float(ans_list_1[i] + ans_list_2[i]), 4)) # use float and then apply round function
    if ans_list == []:
        return [0.0]

    return ans_list

'''
Notes
- list length = len(list)
- to find the maximum of values, use max() function
- to make new list, itialize empty list and use append() method to add elements in for loop
- list index can't be bigger then the length of list - 1
- to use index and value in the list, use enumerate() function
    - for i, coeff in enumerate(f_coeffs):
        # i is the index, coeff is the value at that index
- if i know the length of the list and want to initialize it, use [0] * length
    - ans_list = [0] * ans_length
- to add list with different length, compare the length of two list and increase the length of the shorter list
- when multiplying two polynomials, the answer list will have a length of (f_length + g_length - 1)
- if you want to element-wise add two lists, you should use a for loop
- if you want to make the output rounded to 4 decimal places, use round() function
    - round(value, 4)
'''

'''
Gemini Solution - 1

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials using pure Python.
    """

    # 1. Compute the product of the two polynomials (f * g)
    fg_coeffs = [0.0] * (len(f_coeffs) + len(g_coeffs) - 1)
    for i, f in enumerate(f_coeffs):
        for j, g in enumerate(g_coeffs):
            fg_coeffs[i + j] += f * g

    # 2. Compute the derivative of the product and round to 4 decimal places
    ans_list = []
    for i in range(1, len(fg_coeffs)):
        ans_list.append(round(float(fg_coeffs[i] * i), 4))

    # 3. Remove trailing zeros to ensure clean output
    while ans_list and ans_list[-1] == 0.0:
        ans_list.pop()

    return ans_list if ans_list else [0.0]

Gemini Solution - 2 (use numpy library)

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    if not f_coeffs or not g_coeffs:
        return [0.0]
        
    # np.convolve is used to get the coefficients when multiplying two polynomials
    fg_coeffs = np.convolve(f_coeffs, g_coeffs)
    
    ans_list = [round(float(c * i), 4) for i, c in enumerate(fg_coeffs)][1:]

    while ans_list and ans_list[-1] == 0.0:
        ans_list.pop()
        
    return ans_list if ans_list else [0.0]

'''