import sys
sys.path.append('../')
from src.Logic.Parser import *

val = {'A': np.float64(0.7), 'B': np.float64(0.5)}

Parser = Parser(Lukasiewicz())

result, depth, _ = Parser.generate_ast("(A⊙B)")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⊙, 0), (A, 1), (B, 1)]")
print(depth)

result = Parser.evaluate_formula(result, val)
print("Example 1: " + str(result) + " Expected Result: 0.2") 

result, depth, _ = Parser.generate_ast("(A⇒B)")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⇒, 0), (A, 1), (B, 1)]")
print(depth)

result = Parser.evaluate_formula(result, val)
print("Example 2: " +  str(result) + " Expected Result: 0.8") 

result, depth, _ = Parser.generate_ast("(¬A)")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(¬, 0), (A, 1)]")
print(depth)

result = Parser.evaluate_formula(result, val)
print("Example 3: " + str(result) + " Expected Result: 0.3") 

result, depth, _ = Parser.generate_ast("(A⇒(B⊕(¬A)))")
print("Level Order Traversal: " +  str(Tree.level_order_traversal(result)) + " Expected Result: [(⇒, 0), (A, 1), (⊕, 1), (B, 2), (¬, 2), (A, 3)]")
print(depth)

result = Parser.evaluate_formula(result, val)
print("Example 4: " + str(result) + " Expected Result: 1")

result, depth, _ = Parser.generate_ast("(A⊙(B⇒(¬A)))")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⊙, 0), (A, 1), (⇒, 1), (B, 2), (¬, 2), (A, 3)]")
print(depth)

result = Parser.evaluate_formula(result, val)
print("Example 5: " + str(result) + " Expected Result: 0.5")

# ----------------------------------------------------------- #

tree_no_degs, depth, _ = Parser.generate_ast("(A⊙A)")
tree_with_degs, depth, _ = Parser.generate_ast_with_degs("(A⊙A)")

print(Parser.evaluate_formula(tree_with_degs, val) - Parser.evaluate_formula(tree_no_degs, val))

tree_no_degs, depth, _ = Parser.generate_ast("((A⊙A)⊙(A⊙B))")
tree_with_degs, depth, _ = Parser.generate_ast_with_degs("((A⊙B)⊙(A⊙B))")

def print_tree(Node: Tree.Node):
    if Node == None:
        return
    else:
        print(Node)
        print(Node.data)
        print_tree(Node.right)
        print_tree(Node.left)

print_tree(tree_with_degs)

print(Parser.evaluate_formula(tree_with_degs, val) - Parser.evaluate_formula(tree_no_degs, val))


# ----------------------------------------------------------- #

formula =  ("((x2∧x1)V(x3∧x1))")

print("Formula gotten:" + Parser.minimize_smaller_formula(formula, variables = ["x1", "x2", "x3"]) + " Formula expected: (x1∧(x2Vx3))")

formula = ("(0⊙x)")

print(Parser.minimize_trivial(formula) + " Expected Result: 0")

formula = ("(1⊙x)")

print(Parser.minimize_trivial(formula) + " Expected Result: x")

formula = ("(x⊙0)")

print(Parser.minimize_trivial(formula) + " Expected Result: 0")

formula = ("(x⊙1)")

print(Parser.minimize_trivial(formula) + " Expected Result: x")

formula = "(x⊕1)"

print(Parser.minimize_trivial(formula) + " Expected Result: 1")

formula = "(x⊕0)"

print(Parser.minimize_trivial(formula) + " Expected Result: x")

formula = "(1⊕x)"

print(Parser.minimize_trivial(formula) + " Expected Result: 1")

formula = "(0⊕x)"

print(Parser.minimize_trivial(formula) + " Expected Result: x")


# ----------------------------------------------------------- #

formula = "(x1⊕(x2⊕x3))"
values = [0, 0.5, 1]

# evaluate both formulas to check if they are equivalent
root, depth, _ = Parser.generate_ast(formula)
result = Parser.evaluate_formula(root, {'x1': 0, 'x2': 0, 'x3': 0.5})
print(result)

root, depth, _ = Parser.generate_ast(Parser.minimize_expresso_mv(formula, values=values))
result = Parser.evaluate_formula(root, {'x1': 0, 'x2': 0, 'x3': 0.5})
print(result)

# ----------------------------------------------------------- #
formula = "(δ_2 ((x2∧x1)V(x3∧x1))"

print("Formula gotten:" + Parser.minimize_delta_formulas(formula, Parser.minimize_smaller_formula, variables = ["x1", "x2", "x3"]) + " Formula expected: (δ_2 ((x1∧(x3Vx2)))")




