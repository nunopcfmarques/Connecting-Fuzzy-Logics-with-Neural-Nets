import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.TLogicToReLU import *

Lukasiewicz = Lukasiewicz()
Godel = Godel()

def construct_lukasiewicz_delta_ReLU(i: int):
    x = np.float64(1 / int(i))
    return ReLUNetwork(
        [torch.tensor([[x]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    )

Lukasiewicz_connectives_to_ReLU = {
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
    "⇒": ReLUNetwork(
        [torch.tensor([[1., -1.]], dtype=torch.float64), torch.tensor([[-1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
    ),
    "": ReLUNetwork(
        [torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "δ": construct_lukasiewicz_delta_ReLU
}

Godel_connectives_to_ReLU = {
    "⊕": ReLUNetwork(
        [torch.tensor([[1., 0.], [-1., 1.]], dtype=torch.float64), torch.tensor([[1., 1.]], dtype=torch.float64)],
        [torch.tensor([0., 0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "⊙": ReLUNetwork(
        [torch.tensor([[1., 0.], [1., -1.]], dtype=torch.float64), torch.tensor([[1., -1.]], dtype=torch.float64)],
        [torch.tensor([0., 0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "": ReLUNetwork(
        [torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
}

LukasiewiczToReLU = LogicToRelu(Lukasiewicz_connectives_to_ReLU, Lukasiewicz)
GodelToReLU = LogicToRelu(Godel_connectives_to_ReLU, Godel)

# Examples
formulas = [
    ("(x⊙y)"),         # x AND y
    ("(x⊙(y⊙z))"),     # x AND (y AND z)
    ("(x⊕y)"),         # x XOR y
    ("(x⊕(y⊕z))"),     # x XOR (y XOR z)
    ("(¬x)"),             # NOT x
    ("((¬x)⊙y)"),        # NOT x AND y
    ("(x⊙((¬y)⊙z))"),   # x AND (NOT y AND z)
    ("δ_10 x")
]

for formula in formulas:
    root, max_depth = Lukasiewicz.generate_ast(formula)
    ReLU = LukasiewiczToReLU.ast_to_ReLU(root, max_depth)
    print("Formula:", formula)
    print("Weights:", ReLU.weights)
    print("Biases:", ReLU.biases)
    print()

atoms = ["w", "x", "y", "z"]

print("Starting Lukasiewicz Tests")
print()
for i in range(0, 50):
    val = {"w": np.float64(np.random.random_sample()), "x": np.float64(np.random.random_sample()), "y": np.float64(np.random.random_sample()), "z": np.float64(np.random.random_sample())}
    formula = Lukasiewicz.random_formula(atoms, ["¬", "⊙", "⊕", "⇒", "δ"], max_depth=10)
    root, max_depth = Lukasiewicz.generate_ast(formula)
    ReLU = LukasiewiczToReLU.ast_to_ReLU(root, max_depth)
    ReLU.construct_layers()

    print(ReLU(LogicToRelu.valuation_to_tensor(val, formula)).item() - Lukasiewicz.evaluate_formula(root, val))

print("All Good for Lukasiewicz")
print()

print("Starting Godel Tests")
print()
for i in range(0, 50):
    val = {"w":np.random.random_sample(), "x": np.random.random_sample(), "y": np.random.random_sample(), "z": np.random.random_sample()}
    formula = Godel.random_formula(atoms, ["⊙", "⊕"], max_depth=10)
    root, max_depth = Godel.generate_ast(formula)
    ReLU = GodelToReLU.ast_to_ReLU(root, max_depth)
    ReLU.construct_layers()

    assert((ReLU(LogicToRelu.valuation_to_tensor(val, formula)).item() - Godel.evaluate_formula(root, val)) == 0)
print("All Good for Godel")
print()

