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
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([0.7], dtype=torch.float64)

ast, depth = Lukasiewicz.generate_ast_with_degs(formula)
ReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))

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

ast, depth = Lukasiewicz.generate_ast_with_degs(formula)
ReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))


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

ast, depth = Lukasiewicz.generate_ast_with_degs(formula)
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

ReLU.construct_layers()
print(ReLU(torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)))

# Step 1 transform into CReLU
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([0.4, 0.5, 0.3], dtype=torch.float64)
print("here 2")

ast, depth = Lukasiewicz.generate_ast(formula)
ReLU.construct_layers()

formula = Lukasiewicz.minimize_formula(ast)
ast, depth = Lukasiewicz.generate_ast(formula)
print(formula)

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))

print("")
print("")

ReLU = ReLUNetwork (
    [torch.tensor([[.5, -0.5, 0.5],[.5, -0.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

lcm = ReLU.get_general_lcm()

ReLU.transform_rational_to_int(lcm)

# Step 1 transform into CReLU
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))

# Step 2 generate formula
formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)

# Step 3 compare outputs
input_tensor = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)

ast, depth = Lukasiewicz.generate_ast_with_degs(formula)
ReLU.construct_layers()

print(ReLU(input_tensor))
print(Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))
print(ReLU(input_tensor) - (Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor))))

print("")
print("")

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

ast, depth = Lukasiewicz.generate_ast_with_degs(formula)

formula = Lukasiewicz.minimize_formula(ast)
print(formula)

ast, depth = Lukasiewicz.generate_ast_with_degs(formula)

ReLU.construct_layers()
CReLU.construct_layers()

print(ReLU(input_tensor) - Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))



# NETWORK SCALING TESTS!!!

ReLU = ReLUNetwork(
    [torch.tensor([[0.5, -0.5, 0.5], [0.5, -0.5, -0.5]], dtype=torch.float64), 
     torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

ReLU2 = ReLUNetwork(
    [torch.tensor([[0.5, -0.5, 0.5], [0.5, -0.5, -0.5]], dtype=torch.float64), 
     torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

lcm = ReLU.get_general_lcm()
ReLU.transform_rational_to_int(lcm)

print(ReLU.weights)

ReLU.construct_layers()
ReLU2.construct_layers()

ReLU.construct_layers

test_input = torch.tensor([[0.1, 0.3, 1.0]], dtype=torch.float64)

original_output = ReLU2(test_input)

transformed_output = ReLU(test_input)

print("Original Output:", original_output)
print("Transformed Output (scaled):", transformed_output)

ReLU3 = ReLUNetwork(
    [torch.tensor([[0.3, -0.2], [0.5, 0.4]], dtype=torch.float64), 
     torch.tensor([[1.0, -1.0]], dtype=torch.float64)],
    [torch.tensor([0.1, -0.3], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64)]
)

ReLU4 = ReLUNetwork(
    [torch.tensor([[0.3, -0.2], [0.5, 0.4]], dtype=torch.float64), 
     torch.tensor([[1.0, -1.0]], dtype=torch.float64)],
    [torch.tensor([0.1, -0.3], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64)]
)

lcm = ReLU3.get_general_lcm()
ReLU3.transform_rational_to_int(lcm)

ReLU3.construct_layers()
ReLU4.construct_layers()

test_input = torch.tensor([[0.6, 0.7]], dtype=torch.float64)

original_output = ReLU4(test_input)

transformed_output = ReLU3(test_input)

print("Original Output:", original_output)
print("Transformed Output (scaled):", transformed_output)

ReLU5 = ReLUNetwork(
    [
        torch.tensor([[0.2, -0.5, 0.3, 0.7], [0.6, -0.4, -0.1, 0.5], [0.8, -0.2, 0.4, 0.1]], dtype=torch.float64),
        torch.tensor([[1.2, -0.7, 0.5], [0.3, -1.0, 0.8]], dtype=torch.float64),
        torch.tensor([[0.5, -0.3]], dtype=torch.float64)
    ],
    [
        torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        torch.tensor([-0.5, 0.7], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64)
    ]
)

ReLU6 = ReLUNetwork(
    [
        torch.tensor([[0.2, -0.5, 0.3, 0.7], [0.6, -0.4, -0.1, 0.5], [0.8, -0.2, 0.4, 0.1]], dtype=torch.float64),
        torch.tensor([[1.2, -0.7, 0.5], [0.3, -1.0, 0.8]], dtype=torch.float64),
        torch.tensor([[0.5, -0.3]], dtype=torch.float64)
    ],
    [
        torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        torch.tensor([-0.5, 0.7], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64)
    ]
)

lcm = ReLU5.get_general_lcm()
ReLU5.transform_rational_to_int(lcm)

ReLU5.construct_layers()

ReLU6.construct_layers()


test_input = torch.tensor([[0.2, 0.5, 0.7, 1.0]], dtype=torch.float64)

original_output = ReLU6(test_input)

transformed_output = ReLU5(test_input)

print("Original Output:", original_output)
print("Transformed Output (scaled):", transformed_output)