import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from MLPToTLogic import *
from Networks.MLP import *
from Logic.Parser import *
from Logic.TLogic import *


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
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([1], dtype=torch.float64)

ast, depth = generate_ast(formula, Lukasiewicz.connectives)
ReLU.construct_layers()

print(ReLU(input_tensor) - evaluate_formula(ast, tensor_to_assignment(input_tensor), Lukasiewicz))

# 2
ReLU = ReLUNetwork (
    [torch.tensor([[-4],[-4]], dtype=torch.float64), torch.tensor([[1, -1]], dtype=torch.float64)],
    [torch.tensor([2, 1], dtype=torch.float64), torch.tensor([0], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([0.4], dtype=torch.float64)

ast, depth = generate_ast(formula, Lukasiewicz.connectives)
ReLU.construct_layers()

print(ReLU(input_tensor) - evaluate_formula(ast, tensor_to_assignment(input_tensor), Lukasiewicz))


# 3
ReLU = ReLUNetwork(
    [torch.tensor([[-2, 0], [0, 1], [0, -1]], dtype=torch.float64), torch.tensor([[-1, -1, 1]], dtype=torch.float64), torch.tensor([[1]], dtype=torch.float64)],
    [torch.tensor([1, 0, 0], dtype=torch.float64), torch.tensor([1], dtype=torch.float64),  torch.tensor([0], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([0.4, 0.2], dtype=torch.float64)

ast, depth = generate_ast(formula, Lukasiewicz.connectives)
ReLU.construct_layers()

print(ReLU(input_tensor) - evaluate_formula(ast, tensor_to_assignment(input_tensor), Lukasiewicz))

# --------------------- #
#  TESTS FOR RATIONALS  #
# --------------------- #

# 1
ReLU = ReLUNetwork (
    [torch.tensor([[.5, -0.5, 0.5],[.5, -0.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([0.4, 0.5, 0.7], dtype=torch.float64)

ast, depth = generate_ast(formula, Lukasiewicz.connectives)
ReLU.construct_layers()

print(ReLU(input_tensor) - evaluate_formula(ast, tensor_to_assignment(input_tensor), Lukasiewicz))

# 2 
ReLU = ReLUNetwork (
    [torch.tensor([[0.5, -0.5],[.5, -0.5]], dtype=torch.float64), torch.tensor([[0.5, -0.5],[.5, -0.5]], dtype=torch.float64), torch.tensor([[0.2, 0.2]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

# Step 1 transform into CReLU
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([0.2, 0.2], dtype=torch.float64)

ast, depth = generate_ast(formula, Lukasiewicz.connectives)
ReLU.construct_layers()
CReLU.construct_layers()

print(ReLU(input_tensor) - evaluate_formula(ast, tensor_to_assignment(input_tensor), Lukasiewicz))