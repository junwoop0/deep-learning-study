# Problem: Single Neuron with Backpropagation
# URL: https://www.deep-ml.com/problems/25
import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	sam_len = np.size(labels)
	MSE_loss = []
	h_value = features @ initial_weights + initial_bias
	for i in range(epochs):
		activation = 1 / (1 + np.exp(-1 * h_value))
		loss = activation - labels
		MSE =  (np.mean(np.square(loss)))
		MSE_loss.append(MSE)
		grad_1 = np.sum(2 * (loss) * activation * (1 - activation) * features[:,0]) / (sam_len)
		grad_2 = np.sum(2 * (loss) * activation * (1 - activation) * features[:,1]) / (sam_len)
		updated_weights = np.array([initial_weights[0] - (learning_rate * grad_1),
									initial_weights[1] - (learning_rate * grad_2)])
		initial_weights = updated_weights
		grad_bias = np.sum(2 * (loss) * activation * (1 - activation) * 1) / (sam_len)
		updated_bias = initial_bias - (learning_rate * grad_bias)
		initial_bias = updated_bias
		h_value = features @ updated_weights + updated_bias
  
	return np.round(updated_weights, 4).tolist(), np.round(updated_bias, 4), np.round(MSE_loss, 4).tolist()

'''
Notes
- To dot product of two vectors, you can use np.dot() or the @ operator.
- To make mean of vector, you can use np.mean().
- To calculate square of vector, you can use np.square().
- To make a vector to the shape you want, you can use np.reshape().
    - ex) x = np.array([1, 2, 3])
          y = x.reshape(3, 1)  # y is now a column vector
- numpy automatically broadcasts operations to match shapes when possible.

'''

'''
ChatGPT Solution - 1
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	sam_len = np.size(labels)
	weights = initial_weights.astype(float).copy()
	bias = float(initial_bias)
	mse_values = []

	for _ in range(epochs):
		z = features @ weights + bias
		activation = 1 / (1 + np.exp(-z))

		loss = activation - labels
		mse = np.mean(loss ** 2)
		mse_values.append(round(float(mse), 4))

		delta = (2 / sam_len) * loss * activation * (1 - activation)
		grad_weights = features.T @ delta
		grad_bias = np.sum(delta)

		weights = weights - learning_rate * grad_weights
		bias = bias - learning_rate * grad_bias

	return np.round(weights, 4), round(float(bias), 4), mse_values
'''
