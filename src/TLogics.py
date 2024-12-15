import numpy as np
import random
from typing import Any
from src.utils import Tree
import re

'''
TLogic is a class that holds some place holder truth functions that are meant to be used as methods os a subclass TLogic if it does not explicitly use them. 
It also contains the methods to constract the abstract syntax tree and to evaluate the formula.
Additionally it defines the set of connectives and how we can map them to their respective truth functions.
'''

class TLogic:
    connectives = {"¬", "⊙", "⊕", "⇒", "δ", "=", "!", "V" , "∧"}

    connectives_to_truth_function = {
        "¬": "NEG",
        "⊙": "CONJ",
        "⊕": "DISJ",
        "⇒": "IMPLIES",
        "δ": "DELTA",
        "=": "EQUALS",
        "!": "UNEQUALS",
        "V": "WEAK_DISJ",
        "∧": "WEAK_CONJ"
    }
    
    def random_formula(self, atoms: list, choosable_connectives: list, max_depth) -> str:
        if max_depth == 0:
            return random.choice(atoms)

        connective = random.choice(choosable_connectives)
        if connective == "¬":
            return f"(¬{self.random_formula(atoms, choosable_connectives, max_depth - 1)})"
        
        elif connective == "δ": # we leave spaces
            return f"(δ_{random.randint(1, 15)} {self.random_formula(atoms, choosable_connectives, max_depth - 1)})"
        
        elif connective == "":
            return random.choice(atoms)
        else:
            return f"({self.random_formula(atoms, choosable_connectives, max_depth - 1)}{connective}{self.random_formula(atoms, choosable_connectives, max_depth - 1)})"
    

    @staticmethod
    def is_enclosed_in_parentheses(s):
    # Check if the string starts with '(' and ends with ')'
        if s.startswith('(') and s.endswith(')'):
            # Strip outer parentheses and check for balance inside
            inner_content = s[1:-1]
            
            # Use a counter to track parentheses balance
            balance = 0
            for char in inner_content:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                
                # If the balance is negative, it means there's an unmatched closing parenthesis
                if balance < 0:
                    return False
            
            # The final balance should be zero if all parentheses are properly closed
            return balance == 0 and len(inner_content) > 0

        return False

    @staticmethod
    def subdivide_formula(formula: str, correct_parenthesis = True) -> tuple:
        # Takes the parenthesis out of the formula and divides it according to its arity

        if TLogic.is_enclosed_in_parentheses(formula):
            formula = formula[1:-1]

        subformula_count = 0
        for index, character in enumerate(formula):
            if character == "(":
                subformula_count += 1
            elif character == ")":
                subformula_count -= 1
            elif subformula_count == 0 and character in TLogic.connectives:
                if character == "¬":
                    return formula[index + 1:], None, character
                elif character == "δ":
                    i = re.match(r'\d+', formula[index + 2:]).group(0)
                    return formula[index + 3 + len(i):], None, f"{character}{i}" # we need to get δ_i and i is the next character
                elif character == "!":
                    return formula[:index], formula[index + 2:], "!="
                elif character == "=":
                     return formula[:index], formula[index + 2:], "=="
                else:
                    return formula[:index], formula[index + 1:], character

        return formula, None, None
    
    @staticmethod
    def is_atom(formula: str) -> bool:
        # atoms are always a character followed by 0 or more digits
        return re.match(r'[a-zA-Z]\d*', formula)
    
    @staticmethod
    def is_constant(formula: str) -> bool:
        for char in formula:
            if char in Lukasiewicz.connectives:
                return False
        return True
    
    def generate_ast(self, formula: str, depth: int = 0) -> tuple[Tree.Node, int]:
        if self.is_constant(formula):
            node = Tree.Node(formula, depth)
            return node, depth

        else:
            l_formula, r_formula, connective = self.subdivide_formula(formula)
            root = Tree.Node(connective, depth)

            if connective[0] == "¬" or connective[0] == "δ":
                left_node, max_depth = self.generate_ast(l_formula, depth + 1)
                root.left = left_node
                left_node.parent = root
            else:
                left_node, left_max_depth = self.generate_ast(l_formula, depth + 1)
                right_node, right_max_depth = self.generate_ast(r_formula, depth + 1)
                root.left = left_node
                root.right = right_node

                left_node.parent = root
                right_node.parent = root

                # Determine the maximum depth between the left and right subtrees
                max_depth = max(left_max_depth, right_max_depth)

            return root, max_depth
        
    def generate_formula_from_ast(self, root: Tree.Node)-> str:
        if root.left == None:
           return root.data
            
        else:
            if root.data[0] == "¬":
                formula =  f'(¬{self.generate_formula_from_ast(root.left)})'
            elif root.data[0] == "δ":
                formula = f'(δ_{root.data[1:]} {self.generate_formula_from_ast(root.left)})'
            else:
                formula =  f'({self.generate_formula_from_ast(root.left)}{root.data[0]}{self.generate_formula_from_ast(root.right)})'
        
            return formula


    def minimize_formula(self, root: Tree.Node) -> str:
        def dfs_reduction(node):
            if node is None:
                return
            
            changes_made = False

            if node.data[0] == "⊕":
                if node.left.data == "0":
                    if node.parent:
                        replace_node(node, node.right)
                    else:
                        node = node.right
                    changes_made = True

                elif node.right.data == "0":
                    if node.parent:
                        replace_node(node, node.left)
                    else:
                        node = node.left
                    changes_made = True

                elif node.left.data == "1":
                    if node.parent:
                        replace_node(node, node.left)
                    else:
                        node = node.left

                elif node.right.data == "1":
                    if node.parent:
                        replace_node(node, node.right)
                    else:
                        node = node.right

            elif node.data[0] == "⊙":
                if node.left.data == "0":
                    if node.parent:
                        replace_node(node, node.left)
                    else:
                        node = node.left

                elif node.right.data == "0":
                    if node.parent:
                        replace_node(node, node.right)
                    else:
                        node = node.right

                elif node.left.data == "1":
                    if node.parent:
                        replace_node(node, node.right)
                    else:
                        node = node.right

                elif node.right.data == "1":
                    if node.parent:
                        replace_node(node, node.left)
                    else:
                        node = node.left

            elif node.data[0] == "δ":
                if node.left.data == "0":
                    if node.parent:
                        replace_node(node, node.left)
                    else:
                        node = node.left

            if not changes_made:
                if node.left:
                    dfs_reduction(node.left)
                if node.right:
                    dfs_reduction(node.right)

            return node

        def replace_node(node, replacement):
            parent = node.parent
            if parent.left == node:
                parent.left = replacement
            elif parent.right == node:
                parent.right = replacement
            replacement.parent = parent


        old_formula = ""
        while True:
            new_formula = self.generate_formula_from_ast(root)
            if new_formula == old_formula:
                break
            old_formula = new_formula
            root = dfs_reduction(root)

        return self.generate_formula_from_ast(root)

    
    def generate_ast_with_degs(self, formula:str, subformula_to_node={}, depth=0) -> tuple[Tree.Node, int]:
        if formula in subformula_to_node:
            return subformula_to_node[formula]
        
        elif self.is_atom(formula):
            return Tree.Node(formula, depth), depth
        
        elif self.is_constant(formula):
            return Tree.Node(formula, depth), depth
        
        else:
            l_formula, r_formula, connective = self.subdivide_formula(formula)
            root = Tree.Node(connective, depth)
            if connective[0] == "¬":
                left_node, max_depth = self.generate_ast_with_degs(l_formula, subformula_to_node, depth + 1)
                root.left = left_node

            elif connective[0] == "δ":
                left_node, max_depth = self.generate_ast_with_degs(l_formula, subformula_to_node, depth + 1)
                root.left = left_node

            else:
                left_node, left_max_depth = self.generate_ast_with_degs(l_formula, subformula_to_node, depth + 1)
                right_node, right_max_depth = self.generate_ast_with_degs(r_formula, subformula_to_node, depth + 1)
                root.left = left_node
                root.right = right_node
                max_depth = max(left_max_depth, right_max_depth)

            subformula_to_node[formula] = (root, max_depth)
            return subformula_to_node[formula]
    
    def get_function_name(self, connective: str) -> Any:
        return getattr(self, self.connectives_to_truth_function[connective[0]])
    
    def evaluate_formula(self, root: Tree.Node, val: dict) -> np.float64:
        if root.left == None:
            if root.data in val:
                return val[root.data]
            else:
                return float(root.data)
            
        else:
            function = self.get_function_name(root.data)

            if root.data[0] == "¬":
                eval = function(self.evaluate_formula(root.left, val))
            elif root.data[0] == "δ":
                eval = function(root.data[1:], self.evaluate_formula(root.left, val))
            else:
                eval = function(self.evaluate_formula(root.left, val), self.evaluate_formula(root.right, val))
        
            return eval

# Here the multiplicative connectives collapse! meaning ∧ is ⊙ and ∨ is ⊕
class Godel(TLogic):
    def IMPLIES(self, x: np.float64, y: np.float64) -> np.float64:
        return y if x > y else np.float64(1)
    
    #v(A∧B) = min(v(A),v(B))
    def CONJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.minimum(x, y)

    #v(A∨B) = max(v(A),v(B))
    def DISJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.maximum(x, y)

class Product(TLogic):
    @staticmethod
    def CONJ(x: np.float64, y: np.float64) -> np.float64:
        pass

    @staticmethod
    def DISJ(x: np.float64, y: np.float64) -> np.float64:
        pass

    @staticmethod
    def NEG(x: np.float64) -> np.float64:
        pass


class Lukasiewicz(TLogic):
    #v(A → B) = min(1,1−v(A) +v(B))
    def IMPLIES(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(1, 1 - x + y))

    #v(A⊙B) = max(0,v(A) +v(B)−1)
    def CONJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.maximum(0, x + y - 1))

    #v(A⊕B) = min(1,v(A) +v(B))
    def DISJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(1, x + y))
    
    #v(¬A) = 1−v(A)
    def NEG(self, x: np.float64) -> np.float64:
        return np.float64(1 - x)
    
    def DELTA(self, i: int, x: np.float64 = 1.) -> np.float64:
        return np.float64(x / int(i))
    
    def EQUALS(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(x == y)

    def UNEQUALS(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(x != y)
    
    def WEAK_CONJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(x, y))
    
    def WEAK_DISJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.maximum(x, y))