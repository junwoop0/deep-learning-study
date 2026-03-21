# Problem: Momentum Optimizer
# URL: https://www.deep-ml.com/problems/146

import numpy as np

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the momentum optimizer.
    Uses momentum to accelerate learning in relevant directions and dampen oscillations.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        velocity: Current velocity/momentum term
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    velocity = (momentum * velocity) + (learning_rate * grad)
    parameter -= velocity

    return np.round(parameter, 5), np.round(velocity, 5)


'''
Notes
- Momentum helps to accelerate gradients vectors in the right directions, which leads to faster converging.
'''