import numpy as np

def sigmoid(x):
    """
    Computes the sigmoid function for scalar or vector x
    """
    return 1 / (1 + np.exp(-x))


def sym_sigmoid(x):
    """
    Computes symmetric sigmoid function based on tanh
    Works for scalar or vector input x
    y = a*tanh(s*x)
    y = a - 2a / ( 1 + exp(2*s*x) )
    """
    a = 1.7159
    s = 2/3
    return a - 2*a / (1 + np.exp(2*s*x))


def dev_sym_sigmoid(x):
    """
    Computes the derivative of the symmetrical sigmoid activation function
    """
    a = 1.7159
    s = 2/3
    exponential = np.exp(2*s*x)
    return 4*a*s*exponential / (1 + exponential) ** 2
