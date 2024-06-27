import re
import random
from Logic.TLogic import *
from utils import Tree


# --------------- #
# Utility Methods #
# --------------- #
def is_atom(formula: str) -> bool:
    special_characters = {"¬", "⊙", "⊕", "⇒", "δ", "V", "∧", "(", ")", "T", "⊥"}
    return not any(character in special_characters for character in formula)

def random_formula(atoms: list, connectives: set, max_depth) -> str:
    if max_depth == 0:
        return random.choice(atoms)

    connective = random.sample((connectives), 1)[0]
    if connective == "¬":
        return f"(¬{random_formula(atoms, connectives, max_depth - 1)})"
        
    elif connective == "δ": # we leave spaces
        return f"(δ_{random.randint(1, 15)} {random_formula(atoms, connectives, max_depth - 1)})"
        
    elif connective == "":
        return random.choice(atoms)
    else:
        return f"({random_formula(atoms, connectives, max_depth - 1)}{connective}{random_formula(atoms, connectives, max_depth - 1)})"
        

# ---------------------------- #
# Abstract Syntax Tree Methods #
# ---------------------------- #
def divide_formula(formula: str, connectives: set) -> tuple:
        if formula.startswith("(") and formula.endswith(")"):
            formula = formula[1:-1]

        subformula_count = 0
        for index, character in enumerate(formula):
            if character == "(":
                subformula_count += 1
            elif character == ")":
                subformula_count -= 1
            elif subformula_count == 0 and character in connectives: # finds the connective of the top level formula
                if character == "¬":
                    return formula[index + 1:], None, character
                elif character == "δ":
                    i = re.match(r'\d+', formula[index + 2:]).group(0)
                    return formula[index + 3 + len(i):], None, f"{character}{i}" # we need to get δ_i and i is the next character
                else:
                    return formula[:index], formula[index + 1:], character
        return formula, None, None

def generate_ast(formula: str, connectives: set, depth: int = 0) -> tuple[Tree.Node, int, list[str]]:
    if is_atom(formula):
        return Tree.Node(formula, depth), depth
        
    elif formula == "T" or formula == "⊥":
        return Tree.Node(formula, depth), depth

    else:
        l_formula, r_formula, connective = divide_formula(formula, connectives)
        root = Tree.Node(connective, depth)
        if connective[0] == "¬":
            left_node, max_depth = generate_ast(l_formula, connectives, depth + 1)
            root.left = left_node
        elif connective[0] == "δ":
            left_node, max_depth = generate_ast(l_formula, connectives, depth + 1)
            root.left = left_node
        else:
            left_node, left_max_depth = generate_ast(l_formula, connectives, depth + 1)
            right_node, right_max_depth = generate_ast(r_formula, connectives, depth + 1)
            root.left = left_node
            root.right = right_node
            max_depth = max(left_max_depth, right_max_depth)

        return root, max_depth
    
def generate_ast_with_degs(formula:str, connectives, subformula_to_node={}, depth=0) -> tuple[Tree.Node, int]:
    if formula in subformula_to_node:
        return subformula_to_node[formula]
        
    elif is_atom(formula):
        return Tree.Node(formula, depth), depth
        
    elif formula == "T" or formula == "⊥":
        return Tree.Node(formula, depth), depth
        
    else:
        l_formula, r_formula, connective = divide_formula(formula, connectives)
        root = Tree.Node(connective, depth)
        if connective[0] == "¬":
            left_node, max_depth = generate_ast_with_degs(l_formula, connectives, subformula_to_node, depth + 1)
            root.left = left_node

        elif connective[0] == "δ":
            left_node, max_depth = generate_ast_with_degs(l_formula, connectives, subformula_to_node, depth + 1)
            root.left = left_node

        else:
            left_node, left_max_depth = generate_ast_with_degs(l_formula, connectives, subformula_to_node, depth + 1)
            right_node, right_max_depth = generate_ast_with_degs(r_formula, connectives, subformula_to_node, depth + 1)
            root.left = left_node
            root.right = right_node
            max_depth = max(left_max_depth, right_max_depth)

        subformula_to_node[formula] = (root, max_depth)
        return subformula_to_node[formula]

def evaluate_formula(root: Tree.Node, assignment: dict, TLogic) -> np.float64:
    if root.left == None:
        if root.data in assignment:
            return assignment[root.data]
        else: # Case that it is "T" or "⊥"
            return TLogic.characters_to_truth_function[root.data]()
            
    else:
        function = TLogic.characters_to_truth_function[root.data[0]]

        if root.data[0] == "¬":
            eval = function(evaluate_formula(root.left, assignment, TLogic))
        elif root.data[0] == "δ":
            eval = function(root.data[1:], evaluate_formula(root.left, assignment, TLogic))
        else:
            eval = function(evaluate_formula(root.left, assignment, TLogic), evaluate_formula(root.right, assignment, TLogic))
        
        return eval   

