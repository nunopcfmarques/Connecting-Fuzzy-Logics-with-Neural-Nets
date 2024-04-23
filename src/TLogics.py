import numpy as np
import random
from typing import Any
from src.utils import Tree

class TLogic:
    connectives = {"¬", "⊙", "⊕", "⇒"}

    connectives_to_truth_function = {
        "¬": "NEG",
        "⊙": "CONJ",
        "⊕": "DISJ",
        "⇒": "IMPLIES",
    }

    truth_values_to_truth_function = {
        "T": "TRUE",
        "⊥": "FALSE",
    }

    def IMPLIES(self, x, y):
        pass

    def FALSE(self):
        return 0

    # 0 → 0
    def TRUE(self):
        return self.IMPLIES(self.FALSE(), self.FALSE())

    # ¬A = A → ⊥
    def NEG(self, x):
        return self.IMPLIES(x, self.FALSE())

    # A⊙B = ¬(A → ¬B) 
    def CONJ(self, x, y):
        return self.NEG(self.IMPLIES(x, self.NEG(y)))

    # A⊕B = ¬A → B
    def DISJ(self, x, y):
        return self.IMPLIES(self.NEG(x), y)
    
    @staticmethod
    def random_formula(atoms: list, max_depth=10) -> str:
        choosable_connectives = ["¬", "⊙", "⊕", "⇒", ""]

        if max_depth == 0:
            return random.choice(atoms)

        connective = random.choice(choosable_connectives)
        if connective == "¬":
            return "(¬" + TLogic.random_formula(atoms, max_depth - 1) + ")"
        elif connective == "":
            return random.choice(atoms)
        else:
            return "(" + TLogic.random_formula(atoms, max_depth - 1) + connective + TLogic.random_formula(atoms, max_depth - 1) + ")"
    

    @staticmethod
    def subdivide_formula(formula: str) -> tuple:
        '''
        Takes the parenthesis out of the formula and divides it according to its arity
        '''
        if formula.startswith("(") and formula.endswith(")"):
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
                else:
                    return formula[:index], formula[index + 1:], character

        return formula, None, None
    
    def generate_ast(self, formula: str, depth=0) -> tuple[Tree.Node, int]:
        if len(formula) == 1:
            return Tree.Node(formula, depth), depth

        else:
            l_formula, r_formula, connective = self.subdivide_formula(formula)
            root = Tree.Node(connective, depth)
            if connective == "¬":
                left_node, max_depth = self.generate_ast(l_formula, depth + 1)
                root.left = left_node
            else:
                left_node, left_max_depth = self.generate_ast(l_formula, depth + 1)
                right_node, right_max_depth = self.generate_ast(r_formula, depth + 1)
                root.left = left_node
                root.right = right_node
                max_depth = max(left_max_depth, right_max_depth)

        return root, max_depth
    
    def get_function_name(self, connective: str) -> Any:
        return getattr(self, self.connectives_to_truth_function[connective])
    
    def evaluate_formula(self, root: Tree.Node, val: dict) -> np.float64:
        if root.left == None:
            if val[root.data] in self.truth_values_to_truth_function:
                return getattr(self, self.truth_values_to_truth_function[val[root.data]])()
            else:
                return val[root.data]
        else:
            function = self.get_function_name(root.data)

            if root.data == "¬":
                eval = function(self.evaluate_formula(root.left, val))
            else:
                eval = function(self.evaluate_formula(root.left, val), self.evaluate_formula(root.right, val))
        
        return eval


class Godel(TLogic):
    @staticmethod
    def CONJ(x: np.float64, y: np.float64) -> np.float64:
        pass

    @staticmethod
    def DISJ(x: np.float64, y: np.float64) -> np.float64:
        pass

    @staticmethod
    def NEG(x: np.float64) -> np.float64:
        pass


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
        return np.minimum(np.float64(1), np.float64(1) - x + y)

    #v(A⊙B) = max(0,v(A) +v(B)−1)
    def CONJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.maximum(np.float64(0), x + y - np.float64(1))

    #v(A⊕B) = min(1,v(A) +v(B))
    def DISJ(self, x: np.float64, y: np.float64) -> np.float64:
        return np.minimum(np.float64(1), x + y)
    
    #v(¬A) = 1−v(A)
    def NEG(self, x: np.float64) -> np.float64:
        return np.float64(1) - x