# Problem: Implementing Basic Autograd Operations
# URL: https://www.deep-ml.com/problems/26

import numpy as np

class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		return f"Value(data={self.data}, grad={int(self.grad)})"

	def __add__(self, other):
		if type(other) != Value:
			other = Value(other)
		out = Value(self.data + other.data, (self, other), "+")
		def a():
			self.grad += 1 * out.grad
			other.grad += 1 * out.grad
		out._backward = a
		return out

	def __mul__(self, other):
		if type(other) != Value:
			other = Value(other)
		out = Value(self.data * other.data, (self, other), "*")
		def m():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out._backward = m
		return out

	def relu(self):
		out = Value(np.maximum(0,self.data), (self,), "ReLU")
		def r():
			self.grad += np.where(self.data > 0, 1.0, 0) * out.grad
		out._backward = r
		return out

	def backward(self):
		order = []
		visited = set()

		def search(x):
			if x not in visited:
				visited.add(x)
				for p in x._prev:
					search(p)
				order.append(x)
		search(self)
		self.grad = 1
		for i in reversed(order):
			i._backward()
   
'''
Notes
- def __add__(self, other) is a special method in Python that allows you to define the behavior of the addition operator (+) for instances of a class.
    - for example, c = a + b will call a.__add__(b) if a is an instance of a class that defines the __add__ method.
- to put something in _op = '', you have to use string, so you have to put it in '' or "".
- special method in Python: https://docs.python.org/3/reference/datamodel.html#special-method-names
- To make a tuple with one element, you need to include a comma after the element, like this: (element,).
- In autograd, the value should be accumulated in the backward pass like grad += grad.
- To use backward, you can gather nodes in dfs order and then call backward on each node in reverse order.
- Use reverse() to reverse the order of a list in Python.
'''
