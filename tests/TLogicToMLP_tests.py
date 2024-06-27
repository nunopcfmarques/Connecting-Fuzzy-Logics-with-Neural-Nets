import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from TLogicToMLP import *

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
    root, max_depth = generate_ast(formula, Lukasiewicz.connectives)
    ReLU = ast_to_ReLU(root, max_depth, Lukasiewicz_connectives_to_ReLU)
    print("Formula:", formula)
    print("Weights:", ReLU.weights)
    print("Biases:", ReLU.biases)
    print()

atoms = ["w", "x", "y", "z"]

print("Starting Lukasiewicz Tests")
print()
for i in range(0, 50):
    assignment = {"w": np.float64(np.random.random_sample()), "x": np.float64(np.random.random_sample()), "y": np.float64(np.random.random_sample()), "z": np.float64(np.random.random_sample())}
    formula = random_formula(atoms, Lukasiewicz.connectives, max_depth=10)
    root, max_depth = generate_ast(formula, Lukasiewicz.connectives)
    ReLU = ast_to_ReLU(root, max_depth, Lukasiewicz_connectives_to_ReLU)
    ReLU.construct_layers()

    print(ReLU(assignment_to_tensor(assignment, formula)).item() - evaluate_formula(root, assignment, Lukasiewicz))

print("Starting Godel Tests")

for i in range(0, 20):
    assignment = {"w": np.float64(np.random.random_sample()), "x": np.float64(np.random.random_sample()), "y": np.float64(np.random.random_sample()), "z": np.float64(np.random.random_sample())}
    formula = random_formula(atoms, {"∧", "V"}, max_depth=10)
    root, max_depth = generate_ast(formula, {"∧", "V"})
    ReLU = ast_to_ReLU(root, max_depth, Godel_connectives_to_ReLU)
    ReLU.construct_layers()

    print(ReLU(assignment_to_tensor(assignment, formula)).item() - evaluate_formula(root, assignment, Godel))