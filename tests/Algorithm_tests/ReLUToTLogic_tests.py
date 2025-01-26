import sys
sys.path.append('../')

from src.Algorithms.ReLUToTLogic import *
from src.Logic.Parser import *


# 1 
print("Test 1")

Parser = Parser(Lukasiewicz())

ReLU = ReLUNetwork(
    [torch.tensor([[2],[2], [2]], dtype=torch.float64), torch.tensor([[1, -2, 1]], dtype=torch.float64)],
    [torch.tensor([0, -1, -2], dtype=torch.float64),  torch.tensor([0], dtype=torch.float64)]
)

formula, _ = ReLU_to_formula(ReLU)

input_tensor = torch.tensor([0.7], dtype=torch.float64)

ast = Parser.generate_ast_with_degs(formula)[0]
ReLU.construct_layers()

print(ReLU(input_tensor) - Parser.evaluate_formula(ast, tensor_to_valuation(input_tensor)))

# 2
print("Test 2")

ReLU = ReLUNetwork (
    [torch.tensor([[-4],[-4]], dtype=torch.float64), torch.tensor([[1, -1]], dtype=torch.float64)],
    [torch.tensor([2, 1], dtype=torch.float64), torch.tensor([0], dtype=torch.float64)]
)

formula, _ = ReLU_to_formula(ReLU)
input_tensor = torch.tensor([0.4], dtype=torch.float64)

ast = Parser.generate_ast_with_degs(formula)[0]
ReLU.construct_layers()

print(ReLU(input_tensor) - Parser.evaluate_formula(ast, tensor_to_valuation(input_tensor)))


# 3
print("Test 3")
ReLU = ReLUNetwork (
    [torch.tensor([[.5, -0.5, 0.5],[.5, -0.5, -0.5]], dtype=torch.float64), torch.tensor([[-1, 1]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([1], dtype=torch.float64)]
)

formula, _ = ReLU_to_formula(ReLU)
ReLU.construct_layers()

input_tensor = torch.tensor([0.4, 0.5, 0.3], dtype=torch.float64)

ast = Parser.generate_ast_with_degs(formula)[0]

print(ReLU(input_tensor) - Parser.evaluate_formula(ast, tensor_to_valuation(input_tensor)))
