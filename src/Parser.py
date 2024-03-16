from src.Logics.TLogic import TLogic
import random
import copy

class LanguageParser:
    connectives = {"¬", "⊙", "⊕", "⇒", "⊥", "T"}

    connectives_to_function_name = {
			"¬": "self.logic.NEG",
            "⊙": "self.logic.CONJ",
			"⊕": "self.logic.DISJ",
			"⇒": "self.logic.IMPLIES",
			"⊥": "self.logic.FALSE",
			"T": "self.logic.TRUE",
	}

    function_name_to_connective = {value: key for key, value in connectives_to_function_name.items()}

    def __init__(self, TLogic: TLogic):
        self.logic = TLogic

    @staticmethod
    def subdivide_formula(formula: str) -> (tuple):
        if formula[0] == "(":
                formula = formula[1:-1]
        subformula_count = 0
        for index, character in enumerate(formula):
            if character == "(":
                subformula_count += 1
            elif character == ")":
                subformula_count -= 1
            elif subformula_count == 0 and character in LanguageParser.connectives:
                if character == "¬":
                    return (formula[index + 1:], None, character)
                else:
                    return (formula[:index], formula[index + 1:], character)

    @staticmethod
    def formula_parser_to_function(formula: str) -> str:
        if len(formula) == 1:
            return "val["+ '"' + formula + '"' + "]"
        
        else:
            l_formula, r_formula, connective = LanguageParser.subdivide_formula(formula)
            function_name = LanguageParser.connectives_to_function_name[connective]

            if function_name == "self.logic.NEG":
                parsed = function_name + "(" + LanguageParser.formula_parser_to_function(l_formula) + ")"

            else:
                parsed = function_name + "(" + LanguageParser.formula_parser_to_function(l_formula) + "," + LanguageParser.formula_parser_to_function(r_formula) + ")"

        return parsed
    
    @staticmethod
    def parse_function_to_formula(str_function: str) -> (tuple):

        function_name = str_function[:str_function.index('(')]
        arguments = str_function[str_function.index('('):]

        if function_name == "self.logic.NEG":
            arguments = arguments[1:-1]
            return (arguments, None, LanguageParser.function_name_to_connective[function_name])
        else:
            arguments = arguments[1:-1]
            subformula_count = 0
            for index, character in enumerate(arguments):
                if character == "(":
                    subformula_count += 1
                elif character == ")":
                    subformula_count -= 1
                elif subformula_count == 0 and character == ",":
                    return (arguments[:index], arguments[index + 1:], LanguageParser.function_name_to_connective[function_name])

    @staticmethod 
    def function_parser_to_formula(str_function: str) -> str:
        if str_function.startswith("val"):
            return str_function[str_function.find('["') + 1: str_function.find('"]')]
        
        else:
            l_str_function, r_str_function, connective = LanguageParser.parse_function_to_formula(str_function)

            if connective == "¬":
                parsed = "(" + connective + LanguageParser.function_parser_to_formula(l_str_function) + ")"
            else:
                parsed = "(" + LanguageParser.function_parser_to_formula(l_str_function) + connective + LanguageParser.function_parser_to_formula(r_str_function) + ")"

        return parsed
    
    @staticmethod
    def contains_symbol(string: str) -> bool:
        for char in string:
            if char in LanguageParser.connectives:
                return True
        return False
    
    @staticmethod
    def formula_to_tree(formula: str) -> tuple[dict, list[str], int]:
        expand_queue = []
        atoms = LanguageParser.atoms(formula)
        depth_to_ReLUs = {0: []}
    
        expand_queue.append((formula, 0))

        deepest_layer = 0

        while set(depth_to_ReLUs[deepest_layer]) != atoms:
            (exp_formula, exp_layer) = expand_queue.pop(0)

            if exp_layer > deepest_layer:
                deepest_layer = exp_layer

            if LanguageParser.contains_symbol(exp_formula):
                subformula1, subformula2, connective = LanguageParser.subdivide_formula(exp_formula)
                expand_queue.append((subformula1, exp_layer + 1))
                if subformula2 is not None:
                    expand_queue.append((subformula2, exp_layer + 1))
            else:
                expand_queue.append((exp_formula, exp_layer + 1))
                connective = exp_formula
            
            if exp_layer in depth_to_ReLUs:
                depth_to_ReLUs[exp_layer].append(connective)
            else:
                depth_to_ReLUs[exp_layer] = [connective]
        
        depth = deepest_layer - 1
        if deepest_layer != 0:     
            input_layer = depth_to_ReLUs.popitem()[1]
        else:
            input_layer = copy.deepcopy(depth_to_ReLUs[0])
            depth = depth + 1
            
        return (depth_to_ReLUs, input_layer, depth)

    @staticmethod
    def atoms(formula: str) -> set:
        atoms = set()
        i = 0
        while i < len(formula):
            character = formula[i]
            if character not in LanguageParser.connectives and character != "(" and character != ")":
                atom = character
                i += 1
                while i < len(formula) and formula[i].isdigit():
                    atom += formula[i]
                    i += 1
                atoms.add(atom)
            else:
                i += 1

        return atoms
    
    @staticmethod
    def freshVariables(formula: str) -> str:

        atoms_count = {}
        atoms = LanguageParser.atoms(formula)

        characters = [character for character in formula]

        for atom in atoms:
            atoms_count[atom] = 0

        for index, character in enumerate(characters):
            if character in atoms:
                atoms_count[character] += 1
                variable = character + str(atoms_count[character])
                characters[index] = variable
        
        return ''.join(characters)
    
    @staticmethod
    def random_formula(atoms: set, max_depth=10) -> str:
        choosable_connectives = ["¬", "⊙", "⊕", "atom"]

        if max_depth == 0:
            return random.choice(list(atoms))

        connective = random.choice(choosable_connectives)
        if connective == "¬":
            return "(¬" + LanguageParser.random_formula(atoms, max_depth - 1) + ")"
        elif connective == "atom":
            return random.choice(list(atoms))
        else:
            return "(" + LanguageParser.random_formula(atoms, max_depth - 1) + connective + LanguageParser.random_formula(atoms, max_depth - 1) + ")"


    def evaluate(self, str_function, val):
        result = eval(str_function)
        return result