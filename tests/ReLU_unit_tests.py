import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.ReLUNetwork import ReLUNetwork

bias_array1 = torch.tensor([1, 1])
bias_array2 = torch.tensor([2])
expected_result = torch.tensor([1, 1, 2])
result = ReLUNetwork.vertically_append_biases(bias_array1, bias_array2)
assert torch.equal(result, expected_result)

weight_matrix1 = torch.tensor([[1, 2], [3, 4]])
weight_matrix2 = torch.tensor([[5, 6]])
expected_result = torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6]])
result = ReLUNetwork.vertically_append_weights(weight_matrix1, weight_matrix2)
assert torch.equal(result, expected_result)

weights1 = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 2], [3, 4]])]
biases1 = [torch.tensor([1, 1]), torch.tensor([1, 1])]
weights2 = [torch.tensor([[5, 6]]), torch.tensor([[5, 6]])]
biases2 = [torch.tensor([2]), torch.tensor([2])]

ReLU1 = ReLUNetwork(weights1, biases1)
ReLU2 = ReLUNetwork(weights2, biases2)

expected_weights = [torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6]]), torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6]])]
expected_biases = [torch.tensor([1, 1, 2]), torch.tensor([1, 1, 2])]

ReLU3 = ReLUNetwork.vertically_append_ReLUs(ReLU1, ReLU2)
for expected, actual in zip(expected_weights, ReLU3.weights):
    assert torch.equal(expected, actual)
for expected, actual in zip(expected_biases, ReLU3.biases):
    assert torch.equal(expected, actual)

weights1 = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])]
biases1 = [torch.tensor([1, 1]), torch.tensor([1])]
weights2 = [torch.tensor([[7, 8], [7, 8]])]
biases2 = [torch.tensor([2])]

ReLU1 = ReLUNetwork(weights1, biases1)
ReLU2 = ReLUNetwork(weights2, biases2)
    
expected_weights = [torch.tensor([[7, 8], [7, 8]]), torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])]
expected_biases = [torch.tensor([2]), torch.tensor([1, 1]), torch.tensor([1])]
    
ReLU3 = ReLUNetwork.horizontally_append_ReLUs(ReLU1, ReLU2)
for expected, actual in zip(expected_weights, ReLU3.weights):
    assert torch.equal(expected, actual)
for expected, actual in zip(expected_biases, ReLU3.biases):
    assert torch.equal(expected, actual)