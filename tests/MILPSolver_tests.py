import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.MILPSolver import *
from src.CReluToTLogic import *
from src.ReLUNetwork import *
from src.CReLUNetwork import *
from copy import deepcopy

# 1
ReLU = ReLUNetwork (
    [torch.tensor([[.5, 0.5, 0.5],[.5, -0.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))

ReLU.construct_layers()

print(ReLU(torch.tensor([0, 0, 0], dtype=torch.float64)))

SolveFormulaMILP(formula)