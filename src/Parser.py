from src.Logics.TLogic import TLogic

class LanguageParser:
    # This is universal
    connective_to_function_name = {
			"¬": "self.logic.NEG",
            "⊙": "self.logic.CONJ",
			"⊕": "self.logic.DISJ",
			"⇒": "self.logic.IMPLIES",
			"⊥": "self.logic.FALSE",
			"T": "self.logic.TRUE",
	}

    function_name_to_connective = {value: key for key, value in connective_to_function_name.items()}

    # L will be a set of characters
    def __init__(self, TLogic: TLogic):
        self.language = TLogic.language
        self.logic = TLogic

    def parse_formula(self, formula) -> (tuple):
        print(formula)
        if formula[0] == "(":
                formula = formula[1:-1]
        subformula_count = 0
        for index, character in enumerate(formula):
            print(character)
            if character == "(":
                subformula_count += 1
            elif character == ")":
                subformula_count -= 1
            elif subformula_count == 0 and character in self.language:
                if character == "¬":
                    print(formula)
                    return (formula[index + 1:], None, self.connective_to_function_name[character])
                else:
                    print(formula)
                    return (formula[:index], formula[index + 1:], self.connective_to_function_name[character])

    def formula_parser(self, formula):
        if len(formula) == 1:
            return "self.logic.val["+ '"' + formula + '"' + "]"
        
        else:
            l_formula, r_formula, function_name = self.parse_formula(formula)

            if function_name == "self.logic.NEG":
                parsed = function_name + "(" + self.formula_parser(l_formula) + ")"

            else:
                parsed = function_name + "(" + self.formula_parser(l_formula) + "," + self.formula_parser(r_formula) + ")"

        return parsed
    
    def parse_function(self, str_function) -> (tuple):

        function_name = str_function[:str_function.index('(')]
        arguments = str_function[str_function.index('('):]

        if function_name == "self.logic.NEG":
            arguments = arguments[1:-1]
            return (arguments, None, self.function_name_to_connective[function_name])
        else:
            arguments = arguments[1:-1]
            subformula_count = 0
            for index, character in enumerate(arguments):
                if character == "(":
                    subformula_count += 1
                elif character == ")":
                    subformula_count -= 1
                elif subformula_count == 0 and character == ",":
                    return (arguments[:index], arguments[index + 1:], self.function_name_to_connective[function_name])
        
    def function_parser(self, str_function):
        if str_function.startswith("self.logic.val"):
            return str_function[str_function.find('["') + 1: str_function.find('"]')]
        
        else:
            l_str_function, r_str_function, connective = self.parse_function(str_function)

            if connective == "¬":
                parsed = "(" + connective + self.function_parser(l_str_function) + ")"
            else:
                parsed = "(" + self.function_parser(l_str_function) + connective + self.function_parser(r_str_function) + ")"

        return parsed
    
    def atoms(self, formula):
        atoms = set()

        for character in formula:
            if character not in self.language and character != "(" and character != ")":
                atoms.add(character)

    def evaluate(self, str_function):
        val = eval(str_function)
        return val