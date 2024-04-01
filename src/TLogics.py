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

    @staticmethod
    def IMPLIES(x: np.float64, y: np.float64) -> np.float64:
        return np.minimum(np.float64(1.0), np.float64(1.0) - x + y).astype(np.float64)

    @staticmethod
    def FALSE() -> np.float64:
        return np.float64(0)

    @staticmethod
    def TRUE() -> np.float64:
        return TLogic.NEG(TLogic.FALSE())

    @staticmethod
    def NEG(x: np.float64) -> np.float64:
        return np.subtract(np.float64(1.0), x).astype(np.float64)

    @staticmethod
    def CONJ(x: np.float64, y: np.float64) -> np.float64:
        return np.maximum(np.float64(0), x + y - np.float64(1)).astype(np.float64)

    @staticmethod
    def DISJ(x: np.float64, y: np.float64) -> np.float64:
        return np.minimum(np.float64(1), x + y).astype(np.float64)
    
    @staticmethod
    def random_formula(atoms: list, max_depth=10) -> str:
        choosable_connectives = ["¬", "⊙", "⊕", ""]

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
    
    @classmethod
    def get_function_name(cls, connective: str) -> Any:
        return getattr(cls, cls.connectives_to_truth_function[connective])
    
    @classmethod
    def generate_ast(cls, formula: str, depth=0) -> tuple[Tree.Node, int]:
        if len(formula) == 1:
            return Tree.Node(formula, depth), depth

        else:
            l_formula, r_formula, connective = cls.subdivide_formula(formula)
            root = Tree.Node(connective, depth)
            if connective == "¬":
                left_node, max_depth = cls.generate_ast(l_formula, depth + 1)
                root.left = left_node
            else:
                left_node, left_max_depth = cls.generate_ast(l_formula, depth + 1)
                right_node, right_max_depth = cls.generate_ast(r_formula, depth + 1)
                root.left = left_node
                root.right = right_node
                max_depth = max(left_max_depth, right_max_depth)

        return root, max_depth
    
    @classmethod
    def evaluate_formula(cls, root: Tree.Node, val: dict) -> np.float64:
        if root.left == None:
            if val[root.data] in cls.truth_values_to_truth_function:
                return getattr(cls, cls.truth_values_to_truth_function[val[root.data]])()
            else:
                return val[root.data]
        else:
            function = cls.get_function_name(root.data)

            if root.data == "¬":
                eval = function(cls.evaluate_formula(root.left, val))
            else:
                eval = function(cls.evaluate_formula(root.left, val), cls.evaluate_formula(root.right, val))
        
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
    @staticmethod
    def IMPLIES(x: np.float64, y: np.float64) -> np.float64:
        return np.minimum(np.float64(1), np.float64(1) - x + y).astype(np.float64)

    @staticmethod
    def CONJ(x: np.float64, y: np.float64) -> np.float64:
        return np.maximum(np.float64(0), x + y - np.float64(1)).astype(np.float64)

    @staticmethod
    def DISJ(x: np.float64, y: np.float64) -> np.float64:
        return np.minimum(np.float64(1), x + y).astype(np.float64)

    @staticmethod
    def NEG(x: np.float64) -> np.float64:
        return np.float64(1) - x.astype(np.float64)