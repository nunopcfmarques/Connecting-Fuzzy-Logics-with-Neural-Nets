import sys
sys.path.append('../')

from src.Algorithms.TLogicToReLU import *
from src.Logic.Parser import *

atoms = ["w", "x", "y", "z"]

print("Starting Lukasiewicz Tests")
print()
prser = Parser(Lukasiewicz())
for i in range(0, 50):
    assignment = {"w": np.float64(np.random.random_sample()), "x": np.float64(np.random.random_sample()), "y": np.float64(np.random.random_sample()), "z": np.float64(np.random.random_sample())}
    connectives = ['¬', '⊙', '⊕', '∧', 'V', '⇒', 'δ']
    formula = prser.random_formula(atoms, connectives, max_depth=10)
    root, max_depth, _ = prser.generate_ast(formula)
    ReLU = ast_to_ReLU(root, max_depth, Lukasiewicz_connectives_to_ReLU)
    ReLU.construct_layers()

    print(ReLU(assignment_to_tensor(assignment, formula)).item() - prser.evaluate_formula(root, assignment))

prser = Parser(Godel())
print()
print("Starting Godel Tests")

for i in range(0, 50):
    assignment = {"w": np.float64(np.random.random_sample()), "x": np.float64(np.random.random_sample()), "y": np.float64(np.random.random_sample()), "z": np.float64(np.random.random_sample())}
    connectives = ['V', '∧']
    formula = prser.random_formula(atoms, connectives, max_depth=10)
    root, max_depth, _ = prser.generate_ast(formula)
    ReLU = ast_to_ReLU(root, max_depth, Godel_connectives_to_ReLU)
    ReLU.construct_layers()

    print(ReLU(assignment_to_tensor(assignment, formula)).item() - prser.evaluate_formula(root, assignment))

