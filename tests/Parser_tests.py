import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from Logic.Parser import *
from Logic.TLogic import *

assignment = {'A': np.float64(0.7), 'B': np.float64(0.5)}

Lukasiewicz = Lukasiewicz()

result, depth = generate_ast("(A⊙B)", Lukasiewicz.connectives)
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⊙, 0), (A, 1), (B, 1)]")
print(depth)

result = evaluate_formula(result, assignment, Lukasiewicz)
print("Example 1: " + str(result) + " Expected Result: 0.2") 

result, depth = generate_ast("(A⇒B)", Lukasiewicz.connectives)
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⇒, 0), (A, 1), (B, 1)]")
print(depth)

result = evaluate_formula(result, assignment, Lukasiewicz)
print("Example 2: " +  str(result) + " Expected Result: 0.8") 

result, depth = generate_ast("(¬A)", Lukasiewicz.connectives)
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(¬, 0), (A, 1)]")
print(depth)

result = evaluate_formula(result, assignment, Lukasiewicz)
print("Example 3: " + str(result) + " Expected Result: 0.3") 

result, depth = generate_ast("(A⇒(B⊕(¬A)))", Lukasiewicz.connectives)
print("Level Order Traversal: " +  str(Tree.level_order_traversal(result)) + " Expected Result: [(⇒, 0), (A, 1), (⊕, 1), (B, 2), (¬, 2), (A, 3)]")
print(depth)

result = evaluate_formula(result, assignment, Lukasiewicz)
print("Example 4: " + str(result) + " Expected Result: 1")

result, depth = generate_ast("(A⊙(B⇒(¬A)))", Lukasiewicz.connectives)
print("Level Order Traversal: " + str(Tree.level_order_traversal(result)) + " Expected Result: [(⊙, 0), (A, 1), (⇒, 1), (B, 2), (¬, 2), (A, 3)]")
print(depth)

result = evaluate_formula(result, assignment, Lukasiewicz)
print("Example 5: " + str(result) + " Expected Result: 0.5")

# ----------------------------------------------------------- #

tree_no_degs, depth = generate_ast("(A⊙A)", Lukasiewicz.connectives)
tree_with_degs, depth = generate_ast_with_degs("(A⊙A)", Lukasiewicz.connectives)

print(evaluate_formula(tree_with_degs, assignment, Lukasiewicz) - evaluate_formula(tree_no_degs, assignment, Lukasiewicz))

tree_no_degs, depth = generate_ast("((A⊙A)⊙(A⊙B))", Lukasiewicz.connectives)
tree_with_degs, depth = generate_ast_with_degs("((A⊙B)⊙(A⊙B))", Lukasiewicz.connectives)

def print_tree(Node: Tree.Node):
    if Node == None:
        return
    else:
        print(Node)
        print(Node.data)
        print_tree(Node.right)
        print_tree(Node.left)

print_tree(tree_with_degs)

print(evaluate_formula(tree_with_degs, assignment, Lukasiewicz) - evaluate_formula(tree_no_degs, assignment, Lukasiewicz))