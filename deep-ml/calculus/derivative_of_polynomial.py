# Problem: Derivative of a Polynomial
# URL: https://www.deep-ml.com/problems/116

def poly_term_derivative(c: float, x: float, n: float) -> float:
    ans = c * n * x ** (n-1)
    return ans

'''
Notes
- A^B should be written as A ** B in Python.
- '-> float:' indicates that the function returns a float value.
'''