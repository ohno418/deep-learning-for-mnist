# This is sample neural network:
#   2 inputs, 3 hidden layers, 2 outputs.

import numpy as np
from activation_functions import sigmoid, identity_function

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    a1 = np.dot(x, network['W1']) + network['b1']
    z1 = sigmoid(a1)
    a2 = np.dot(z1, network['W2']) + network['b2']
    z2 = sigmoid(a2)
    a3 = np.dot(z2, network['W3']) + network['b3']
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)

print('Inputs: ' + str(x))
print('Outputs: ' + str(y))
