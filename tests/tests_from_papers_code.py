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

# The paper gives as test the following ReLU:
ReLU = ReLUNetwork (
    [torch.tensor([[.5, -0.5, 0.5],[.5, -0.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

# And gives us the following CReLU
CReLU = CReLUNetwork(
    [torch.tensor([[ 0.66666667, -0.66666667,  0.66666667], [0.66666667, -0.66666667,  0.66666667]], dtype=torch.float64), torch.tensor([[-1., -1.]], dtype=torch.float64)],
    [torch.tensor([0., -1], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
)

ReLU.construct_layers()
CReLU.construct_layers()

# and outputs the following formula
formula_paper = compose_MV_terms(CReLU, construct_MV_terms_from_paper(CReLU))

print(formula_paper)

ast_paper, depth = Lukasiewicz.generate_ast(formula_paper)

input_tensor = torch.tensor([0.4, 0.5, 0.7], dtype=torch.float64)

# as we can see they don't match???
print(ReLU(input_tensor))
print(CReLU(input_tensor))
print(Lukasiewicz.evaluate_formula(ast_paper, tensor_to_valuation(input_tensor)))

# My implementation gives:
CReLU = transform_ReLU_to_CReLU(deepcopy(ReLU))
CReLU.construct_layers()

formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))

ast, depth = Lukasiewicz.generate_ast(formula)

print(ReLU(input_tensor))
print(CReLU(input_tensor))
print(Lukasiewicz.evaluate_formula(ast, tensor_to_valuation(input_tensor)))

formula_paper = compose_MV_terms(CReLU, construct_MV_terms_from_paper(CReLU))

ast_paper, depth = Lukasiewicz.generate_ast(formula_paper)
print(Lukasiewicz.evaluate_formula(ast_paper, tensor_to_valuation(input_tensor)))

# I guess their implementation doesn't work... (both step 1 and step 2)