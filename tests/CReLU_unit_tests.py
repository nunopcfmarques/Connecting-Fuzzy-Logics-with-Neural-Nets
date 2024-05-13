import sys
import os
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.CReLUNetwork import *

ReLUNetwork = ReLUNetwork(
    [torch.tensor([[2.0, -0.5],[.5, -0.5]], dtype=torch.float64), torch.tensor([[2.0, -0.5],[.5, -0.5]], dtype=torch.float64), torch.tensor([[2.0, -0.5],[.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

CReLUNetwork = transformReLUToCReLU(deepcopy(ReLUNetwork))

print(CReLUNetwork.weights)
print(CReLUNetwork.biases)

ReLUNetwork.construct_layers()
CReLUNetwork.construct_layers()

print(ReLUNetwork.layers)
print(CReLUNetwork.layers)

print(ReLUNetwork(torch.tensor([0.5, 1], dtype=torch.float64)))
print(CReLUNetwork(torch.tensor([0.5, 1], dtype=torch.float64)))