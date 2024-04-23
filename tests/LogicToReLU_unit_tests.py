import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.LogicToReLU import *

connective_to_ReLU = {
    "⊙": ReLUNetwork(
        [torch.tensor([[1., 1.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([-1.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "¬": ReLUNetwork(
        [torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[-1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
    ),
    "⊕": ReLUNetwork(
        [torch.tensor([[-1., -1.]], dtype=torch.float64), torch.tensor([[-1.]], dtype=torch.float64)],
        [torch.tensor([1.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
    ),
    "": ReLUNetwork(
        [torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
}

LukasiewiczToReLU = LogicToRelu(connective_to_ReLU, Lukasiewicz())

# Examples
formulas = [
    ("(x⊙y)"),         # x AND y
    ("(x⊙(y⊙z))"),     # x AND (y AND z)
    ("(x⊕y)"),         # x XOR y
    ("(x⊕(y⊕z))"),     # x XOR (y XOR z)
    ("(¬x)"),             # NOT x
    ("((¬x)⊙y)"),        # NOT x AND y
    ("(x⊙((¬y)⊙z))"),   # x AND (NOT y AND z)
]

for formula in formulas:
    root, max_depth = TLogic.generate_ast(formula)
    ReLU = LukasiewiczToReLU.ast_to_ReLU(root, max_depth)
    print("Formula:", formula)
    print("Weights:", ReLU.weights)
    print("Biases:", ReLU.biases)
    print()

atoms = ["w", "x", "y", "z"]
for i in range(0, 1000):
    val = {"w":np.random.random_sample(), "x": np.random.random_sample(), "y": np.random.random_sample(), "z": np.random.random_sample()}
    formula = TLogic.random_formula(atoms)
    root, max_depth = TLogic.generate_ast(formula)
    ReLU = LukasiewiczToReLU.ast_to_ReLU(root, max_depth)
    ReLU.construct_layers()

    print(ReLU(LogicToRelu.valuation_to_tensor(val, formula)).item() - TLogic.evaluate_formula(root, val))