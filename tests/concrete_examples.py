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
from src.ExpressoMVParser import *

XOR_Relu = ReLUNetwork(
    weights=[
        torch.tensor([[1, -1], [-1, 1]], dtype=torch.float64),
        torch.tensor([[1, 1]], dtype=torch.float64)
    ],
    biases=[
        torch.tensor([0, 0], dtype=torch.float64),
        torch.tensor([0], dtype=torch.float64)
    ]
)

CReLU = transform_ReLU_to_CReLU(deepcopy(XOR_Relu))

CReLU.construct_layers()

formula = compose_MV_terms(CReLU, construct_MV_terms(CReLU))
print(formula)


tlogic = Lukasiewicz()

variables = [f'x{i+1}' for i in range(2)]

values = [0, 1]
value_to_binary_vector, binary_vector_to_value = BinaryVectorEncoder(values)
ToExpressoMVParser(formula, tlogic, variables, values, value_to_binary_vector)
normal_formula = (FromExpressoMVParser(binary_vector_to_value))
print(normal_formula)

assignment = {"x1": 1, "x2": 1}
ast, _ = tlogic.generate_ast_with_degs(formula)
output = tlogic.evaluate_formula(ast, assignment)

print(output)