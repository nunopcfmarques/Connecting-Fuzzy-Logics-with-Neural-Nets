import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.CReluToTLogic import *
from src.ReLUNetwork import *
from src.CReLUNetwork import *
from src.TLogics import *
from src.TLogicToReLU import * 

Lukasiewicz = Lukasiewicz()


# -------------------- #
#  TESTS FOR INTEGERS  #
# -------------------- #

# 1
ReLU = ReLUNetwork(
    [torch.tensor([[2],[2], [2]], dtype=torch.float64), torch.tensor([[1, -2, 1]], dtype=torch.float64)],
    [torch.tensor([0, -1, -2], dtype=torch.float64),  torch.tensor([0], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transformReLUToCReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))

# Step 3 compare outputs
input_tensor = torch.tensor([0.7], dtype=torch.float64)

atoms = []
ast, depth = Lukasiewicz.generate_ast(formula, atoms)
ReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))

# 2
ReLU = ReLUNetwork (
    [torch.tensor([[-4],[-4]], dtype=torch.float64), torch.tensor([[1, -1]], dtype=torch.float64)],
    [torch.tensor([2, 1], dtype=torch.float64), torch.tensor([0], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transformReLUToCReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))

# Step 3 compare outputs
input_tensor = torch.tensor([0.4], dtype=torch.float64)

atoms = []
ast, depth = Lukasiewicz.generate_ast(formula, atoms)
ReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))


# 3
ReLU = ReLUNetwork(
    [torch.tensor([[-2, 0], [0, 1], [0, -1]], dtype=torch.float64), torch.tensor([[-1, -1, 1]], dtype=torch.float64), torch.tensor([[1]], dtype=torch.float64)],
    [torch.tensor([1, 0, 0], dtype=torch.float64), torch.tensor([1], dtype=torch.float64),  torch.tensor([0], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transformReLUToCReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
# Step 3 compare outputs
input_tensor = torch.tensor([0.4, 0.2], dtype=torch.float64)

atoms = []
ast, depth = Lukasiewicz.generate_ast(formula, atoms)
ReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))

# --------------------- #
#  TESTS FOR RATIONALS  #
# --------------------- #

# 1
ReLU = ReLUNetwork (
    [torch.tensor([[.5, -0.5, 0.5],[.5, -0.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transformReLUToCReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
# Step 3 compare outputs
input_tensor = torch.tensor([0.4, 0.5, 0.7], dtype=torch.float64)

atoms = []
ast, depth = Lukasiewicz.generate_ast(formula, atoms)
ReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))

# 2 
ReLU = ReLUNetwork (
    [torch.tensor([[2, -0.5],[.5, -0.5]], dtype=torch.float64), torch.tensor([[2, -0.5],[.5, -0.5]], dtype=torch.float64), torch.tensor([[0.2, 0.2]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transformReLUToCReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))

# Step 3 compare outputs
input_tensor = torch.tensor([0.8, 0.2], dtype=torch.float64)

atoms = []
ast, depth = Lukasiewicz.generate_ast(formula, atoms)
ReLU.construct_layers()
CReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))
