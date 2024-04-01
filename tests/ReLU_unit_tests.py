import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.ReLUNetwork import ReLUNetwork, np

bias_array1 = np.array([1, 1])
bias_array2 = np.array([2])
expected_result = np.array([1, 1, 2])
result = ReLUNetwork.vertically_append_biases(bias_array1, bias_array2)
assert np.array_equal(result, expected_result)

weight_matrix1 = np.matrix([[1, 2], [3, 4]])
weight_matrix2 = np.matrix([[5, 6]])
expected_result = np.matrix([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6]])
result = ReLUNetwork.vertically_append_weights(weight_matrix1, weight_matrix2)
assert np.array_equal(result, expected_result)

weights1 = [np.matrix([[1, 2], [3, 4]]), np.matrix([[1, 2], [3, 4]])]
biases1 = [np.array([1, 1]), np.array([1, 1])]
weights2 = [np.matrix([[5, 6]]), np.matrix([[5, 6]])]
biases2 = [np.array([2]), np.array([2])]

ReLU1 = ReLUNetwork(weights1, biases1)
ReLU2 = ReLUNetwork(weights2, biases2)

expected_weights = [np.matrix([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6]]), np.matrix([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6]])]
expected_biases = [np.array([1, 1, 2]), np.array([1, 1, 2])]

ReLU3 = ReLUNetwork.compose_vertically(ReLU1, ReLU2)
for expected, actual in zip(expected_weights, ReLU3.weights):
    assert np.array_equal(expected, actual)
for expected, actual in zip(expected_biases, ReLU3.biases):
    assert np.array_equal(expected, actual)

weights1 = [np.matrix([[1, 2], [3, 4]]), np.matrix([[5, 6]])]
biases1 = [np.array([1, 1]), np.array([1])]
weights2 = [np.matrix([[7, 8], [7, 8]])]
biases2 = [np.array([2])]

ReLU1 = ReLUNetwork(weights1, biases1)
ReLU2 = ReLUNetwork(weights2, biases2)
    
expected_weights = [np.matrix([[7, 8], [7, 8]]), np.matrix([[1, 2], [3, 4]]), np.matrix([[5, 6]])]
expected_biases = [np.array([2]), np.array([1, 1]), np.array([1])]
    
ReLU3 = ReLUNetwork.compose_horizontally(ReLU1, ReLU2)
for expected, actual in zip(expected_weights, ReLU3.weights):
    assert np.array_equal(expected, actual)
for expected, actual in zip(expected_biases, ReLU3.biases):
    assert np.array_equal(expected, actual)