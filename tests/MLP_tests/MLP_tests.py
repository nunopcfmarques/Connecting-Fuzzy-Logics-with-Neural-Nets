import torch
import sys

sys.path.append('../')
from src.Networks.MLP import *

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

ReLU3 = ReLUNetwork.vertically_append_MLPs(ReLU1, ReLU2)
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
    
ReLU3 = ReLUNetwork.horizontally_append_MLPs(ReLU1, ReLU2)
for expected, actual in zip(expected_weights, ReLU3.weights):
    assert torch.equal(expected, actual)
for expected, actual in zip(expected_biases, ReLU3.biases):
    assert torch.equal(expected, actual)

ReLU = ReLUNetwork (
    [torch.tensor([[.5, -0.5, 0.5],[.5, -0.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)


CReLU = transform_ReLU_to_CReLU(ReLU)

# compare output of CReLU to with output of ReLU3

input_tensor = torch.tensor([0.4, 0.2, 0.6], dtype=torch.float64)
ReLU.construct_layers()
CReLU.construct_layers()

ReLU_output = ReLU.forward(input_tensor)
CReLU_output = CReLU.forward(input_tensor)

print("ReLU Output:", ReLU_output)
print("CReLU Output:", CReLU_output)

