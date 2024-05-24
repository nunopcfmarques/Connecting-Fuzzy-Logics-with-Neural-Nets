import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.TLogics import *

val = {'A': np.float64(0.7), 'B': np.float64(0.5)}

Lukasiewicz = Lukasiewicz()

result, depth = Lukasiewicz.generate_ast("(A⊙B)")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⊙, 0), (A, 1), (B, 1)]")
print(depth)

result = Lukasiewicz.evaluate_formula(result, val)
print("Example 1: " + str(result) + " Expected Result: 0.2") 

result, depth = Lukasiewicz.generate_ast("(A⇒B)")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⇒, 0), (A, 1), (B, 1)]")
print(depth)

result = Lukasiewicz.evaluate_formula(result, val)
print("Example 2: " +  str(result) + " Expected Result: 0.8") 

result, depth = Lukasiewicz.generate_ast("(¬A)")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(¬, 0), (A, 1)]")
print(depth)

result = Lukasiewicz.evaluate_formula(result, val)
print("Example 3: " + str(result) + " Expected Result: 0.3") 

result, depth = Lukasiewicz.generate_ast("(A⇒(B⊕(¬A)))")
print("Level Order Traversal: " +  str(Tree.level_order_traversal(result)) + " Expected Result: [(⇒, 0), (A, 1), (⊕, 1), (B, 2), (¬, 2), (A, 3)]")
print(depth)

result = Lukasiewicz.evaluate_formula(result, val)
print("Example 4: " + str(result) + " Expected Result: 1")

result, depth = Lukasiewicz.generate_ast("(A⊙(B⇒(¬A)))")
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⊙, 0), (A, 1), (⇒, 1), (B, 2), (¬, 2), (A, 3)]")
print(depth)

result = Lukasiewicz.evaluate_formula(result, val)
print("Example 5: " + str(result) + " Expected Result: 0.5")

# ----------------------------------------------------------- #

tree_no_degs, depth = Lukasiewicz.generate_ast("(A⊙A)")
tree_with_degs, depth = Lukasiewicz.generate_ast_with_degs("(A⊙A)")

print(Lukasiewicz.evaluate_formula(tree_with_degs, val) - Lukasiewicz.evaluate_formula(tree_no_degs, val))

tree_no_degs, depth = Lukasiewicz.generate_ast("((A⊙A)⊙(A⊙B))")
tree_with_degs, depth = Lukasiewicz.generate_ast_with_degs("((A⊙B)⊙(A⊙B))")

def print_tree(Node: Tree.Node):
    if Node == None:
        return
    else:
        print(Node)
        print(Node.data)
        print_tree(Node.right)
        print_tree(Node.left)

print_tree(tree_with_degs)

print(Lukasiewicz.evaluate_formula(tree_with_degs, val) - Lukasiewicz.evaluate_formula(tree_no_degs, val))


val = {'x1': np.float64(0.7), 'x2': np.float64(0.5), 'x3': np.float64(0.5)}

tree_with_degs1, depth = Lukasiewicz.generate_ast("(((((0⊙(¬((x1⊙x1)⊕(x1⊙x1))))⊕(x1⊙x1))⊙(((¬((x1⊙x1)⊕(x1⊙x1)))⊕0)⊙(¬((x1⊙x1)⊙(x1⊙x1)))))⊕(x1⊕x1))⊙(((((¬((x1⊙x1)⊕(x1⊙x1)))⊕0)⊙(¬((x1⊙x1)⊙(x1⊙x1))))⊕(x1⊙x1))⊙((¬((x1⊙x1)⊙(x1⊙x1)))⊕0)))")


print(Lukasiewicz.evaluate_formula(tree_with_degs1, val))