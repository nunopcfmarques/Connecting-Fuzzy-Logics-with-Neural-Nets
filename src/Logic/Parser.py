import re
import random
from src.Logic.TLogic import *
from src.utils import Tree
from itertools import product
import os


class Parser():
    def __init__(self, TLogic) -> None:
        self.TLogic = TLogic

    ####################################################################################################
    # Static methods
    ####################################################################################################
    @staticmethod
    def is_atom(formula: str) -> bool:
        # atoms are always a character followed by 0 or more digits
        return re.match(r'[a-zA-Z]\d*', formula)
    
    @staticmethod
    def is_constant(formula: str) -> bool:
        return re.match(r'^(0(\.\d+)?|1(\.0+)?)$', formula)
    
    @staticmethod
    def is_enclosed_in_parentheses(formula: str) -> bool:
        if formula.startswith('(') and formula.endswith(')'):
            inner_content = formula[1:-1]
            
            balance = 0
            for char in inner_content:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                
                if balance < 0:
                    return False
            
            return balance == 0 and len(inner_content) > 0

        return False
    
    @staticmethod
    def binary_vector_encoder(values: list) -> tuple[dict, dict]:
        num_values = len(values)
        num_bits = int(np.ceil(np.log2(num_values)))

        value_to_binary_vector = {}
        binary_vector_to_value = {}

        for i, value in enumerate(values):
            binary_representation = format(i, f'0{num_bits}b')  # Convert index to binary string with leading zeros
            binary_vector = ''.join([str(bit) for bit in binary_representation])  # Convert binary string to list of bits
            value_to_binary_vector[value] = binary_vector
            binary_vector_to_value[binary_vector] = value

        return value_to_binary_vector, binary_vector_to_value
    
    @staticmethod
    def from_expresso_mv_parser(binary_vector_to_value: dict) -> str:
        num_bits = len(next(iter(binary_vector_to_value.keys())))
        clauses = []

        with open('out.pla', 'r') as espresso_file:
                for _ in range(3): # ignore headers
                    next(espresso_file)
                
                for line in espresso_file:
                    line = line.strip()

                    if line == ".e": # ignore last line
                        break

                    aux = line.split(" ")
                    inputs, output = aux[0], aux[1]
                    inputs_split = [inputs[i:i+num_bits] for i in range(0, len(inputs), num_bits)]
                    
                    clause = "("
                    input_conditions = []
                    
                    for index, inp in enumerate(inputs_split):
                        if "-" in inp:
                            pattern = inp.replace("-", ".")
                            
                            unmatching_values = set(value for key, value in binary_vector_to_value.items() if not re.match(f'^{pattern}$', key))
                            matching_values = set(value for key, value in binary_vector_to_value.items() if re.match(f'^{pattern}$', key))
                            
                            if len(unmatching_values) < len(matching_values):
                                condition = "∧".join(f"(x{index+1}!{val})" for val in unmatching_values)
                                input_conditions.append(condition)
                            else:
                                condition = "∨".join(f"(x{index+1}={val})" for val in matching_values)
                                input_conditions.append(f"({condition})")
                        else:
                            input_conditions.append(f"(x{index+1}={binary_vector_to_value[inp]})")
                    
                    clause += "∧".join(input_conditions)
                    
                    clause += f"∧{binary_vector_to_value[output]})"
                    clauses.append(clause)

        final_clause = "V".join(clauses)
    
        return final_clause
    
    ####################################################################################################
    # Parsing methods
    ####################################################################################################
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
            
    def subdivide_formula(self, formula: str) -> tuple:
        if self.is_enclosed_in_parentheses(formula):
            formula = formula[1:-1]

        subformula_count = 0
        for index, character in enumerate(formula):
            if character == "(":
                subformula_count += 1
            elif character == ")":
                subformula_count -= 1
            elif subformula_count == 0 and character in self.TLogic.connectives:
                if character == "¬":
                    return formula[index + 1:], None, character
                elif character == "δ":
                    i = re.match(r'\d+', formula[index + 2:]).group(0)
                    return formula[index + 3 + len(i):], None, f"{character}{i}" # we need to get δ_i and i is the next character
                else:
                    return formula[:index], formula[index + 1:], character

        return formula, None, None
    
    def generate_ast(self, formula: str, depth: int = 0) -> tuple[Tree.Node, int, int]:
        subformula_count = 1

        if self.is_atom(formula):
            node = Tree.Node(formula, depth)
            return node, depth, subformula_count
        
        if self.is_constant(formula):
            node = Tree.Node(formula, depth)
            return node, depth, subformula_count

        else:
            l_formula, r_formula, connective = self.subdivide_formula(formula)
            root = Tree.Node(connective, depth)

            if connective[0] == "¬" or connective[0] == "δ":
                left_node, max_depth, subformula_count = self.generate_ast(l_formula, depth + 1)
                root.left = left_node
                left_node.parent = root

                if not self.is_atom(l_formula):
                    subformula_count += 1

            else:
                left_node, left_max_depth, subformula_count = self.generate_ast(l_formula, depth + 1)
                right_node, right_max_depth, subformula_count = self.generate_ast(r_formula, depth + 1)
                root.left = left_node
                root.right = right_node

                left_node.parent = root
                right_node.parent = root

                max_depth = max(left_max_depth, right_max_depth)

                if not self.is_atom(l_formula):
                    subformula_count += 1

                if not self.is_atom(r_formula):
                    subformula_count += 1

            return root, max_depth, subformula_count
        
    def generate_ast_with_degs(self, formula: str, subformula_to_node={}, depth: int = 0) -> tuple[Tree.Node, int, int]:
        subformula_count = 1

        if formula in subformula_to_node:
            return subformula_to_node[formula]

        if self.is_atom(formula):
            node = Tree.Node(formula, depth)
            return node, depth, subformula_count
        
        if self.is_constant(formula):
            node = Tree.Node(formula, depth)
            return node, depth, subformula_count

        else:
            l_formula, r_formula, connective = self.subdivide_formula(formula)
            root = Tree.Node(connective, depth)

            if connective[0] == "¬" or connective[0] == "δ":
                left_node, max_depth, subformula_count = self.generate_ast_with_degs(l_formula, subformula_to_node, depth + 1)
                root.left = left_node
                left_node.parent = root

                if not self.is_atom(l_formula):
                    subformula_count += 1
            else:
                left_node, left_max_depth, subformula_count = self.generate_ast_with_degs(l_formula, subformula_to_node, depth + 1)
                right_node, right_max_depth, subformula_count = self.generate_ast_with_degs(r_formula, subformula_to_node, depth + 1)
                root.left = left_node
                root.right = right_node

                left_node.parent = root
                right_node.parent = root

                max_depth = max(left_max_depth, right_max_depth)

                if not self.is_atom(l_formula):
                    subformula_count += 1

                if not self.is_atom(r_formula):
                    subformula_count += 1

            subformula_to_node[formula] = (root, max_depth, subformula_count)
            return subformula_to_node[formula]
        
    def evaluate_formula(self, root: Tree.Node, val: dict) -> np.float64:
        if root.left == None:
            if root.data in val:
                return val[root.data]
            else:
                return float(root.data)
            
        else:
            interpretation = self.TLogic.connectives[root.data[0]].interpretation

            if root.data[0] == "¬":
                eval = interpretation(self.evaluate_formula(root.left, val))
            elif root.data[0] == "δ":
                eval = interpretation(root.data[1:], self.evaluate_formula(root.left, val))
            else:
                eval = interpretation(self.evaluate_formula(root.left, val), self.evaluate_formula(root.right, val))
        
            return eval
        
    ####################################################################################################
    # ExpressoMV methods
    ####################################################################################################
    def formula_to_truth_table(self, formula: str, variables: list[str], values:list, value_to_binary_vector: dict) -> dict:

        num_variables = len(variables)

        ast, _, _ = self.generate_ast(formula)

        possible_combinations = list(product(values, repeat=num_variables))


        truth_table = {}

        for combination in possible_combinations:
            assignment = {variables[i]: combination[i] for i in range(num_variables)}
            output = self.evaluate_formula(ast, assignment)
            truth_table[''.join([value_to_binary_vector[combination[i]] for i in range(num_variables)])] = value_to_binary_vector[output]


        return truth_table
    
    def to_expresso_mv_parser(self, formula: str, variables: list[str], values:list, value_to_binary_vector: dict):

        num_variables = len(variables)

        value_to_binary_vector, _ = Parser.binary_vector_encoder(values)

        size_of_variable = len(next(iter(value_to_binary_vector.values())))

        truth_table = self.formula_to_truth_table(formula, variables, values, value_to_binary_vector)

        with open('espresso_file.pla', 'w+') as espresso_file:
            espresso_file.write(f'.i {num_variables * size_of_variable}\n')
            espresso_file.write(f'.o {size_of_variable}\n') # + 1 because of output
            for inputs in truth_table.keys():
                espresso_file.write(f'{inputs} {truth_table[inputs]}\n')

        os.system("../src/espresso-logic/bin/espresso espresso_file.pla > out.pla")

        return
        

    ####################################################################################################
    # Auxiliary methods for minimization
    ####################################################################################################
    def generate_formulas(self, variables: list[str], depth: int) -> set:
        if depth < 1:
            return set()
            
        if depth == 1:
            return set(variables)

        all_formulas = {d: self.generate_formulas(variables, d) for d in range(1, depth)}
        formulas_at_depth = set()

        binary_connectives = ["⊙", "⊕", "V", "∧"]
        for subformula in all_formulas[depth - 1]:
            formulas_at_depth.add(f"(¬{subformula})")

        for left_depth in range(1, depth):
            right_depth = depth - left_depth
            for left_formula in all_formulas[left_depth]:
                for right_formula in all_formulas[right_depth]:
                    for op in binary_connectives:
                        # Avoid duplicates and trivial (since operations are commutative)
                        new_formula = f"({left_formula}{op}{right_formula})"
                        if (left_formula != right_formula and 
                            f"({right_formula}{op}{left_formula})" not in formulas_at_depth):
                            formulas_at_depth.add(new_formula)

        return formulas_at_depth
    
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
        
    ####################################################################################################
    # Minimization methods
    ####################################################################################################
    def minimize_smaller_formula(self, formula: str, iter: int = 100, **kwargs) -> str:
        worst_score = 1
        variables = kwargs["variables"]
        formula_ast, _, depth = self.generate_ast_with_degs(formula)
        formulas = self.generate_formulas(variables, depth)
        output_formula = ""
        for new_formula in formulas:
            score = 0
            new_formula_ast, _, _ = self.generate_ast_with_degs(new_formula)
            for _ in range(0, iter):
                assignment = {variable: random.random() for variable in variables}
                score += np.abs(self.evaluate_formula(formula_ast, assignment) - self.evaluate_formula(new_formula_ast, assignment)) / iter
            if score < worst_score:
                output_formula = new_formula
                worst_score = score

        return output_formula

    def minimize_trivial(self, formula) -> str:
        root = self.generate_ast(formula)[0]
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
    
    def minimize_expresso_mv(self, formula: str, **kwargs) -> str:
        variables = re.findall(r'\b[a-zA-Z]+\d*\b', formula)
        values = kwargs["values"]
        
        # Remove duplicates
        unique_variables = list(set(variables))
        
        # Sort by the number part of the variable
        sorted_variables = sorted(unique_variables, key=lambda v: (re.match(r'[a-zA-Z]+', v).group(0), int(re.search(r'\d+', v).group(0)) if re.search(r'\d+', v) else float('inf')))

        value_to_binary_vector, binary_vector_to_value = Parser.binary_vector_encoder(values)
        self.to_expresso_mv_parser(formula, sorted_variables, values, value_to_binary_vector)
        normal_formula = Parser.from_expresso_mv_parser(binary_vector_to_value)

        return normal_formula
    
    def minimize_delta_formulas(self, formula: str, minimize_func, **kwargs) -> str:
        index = 0
        while index < len(formula):
            if formula[index] == "δ":
                i = re.match(r'\d+', formula[index + 2:]).group(0)
                if formula[index + 3 + len(i)] == "(":
                    start_index = index + 3 + len(i)
                    stack = 0
                    for j in range(start_index, len(formula)):
                        if formula[j] == '(':
                            stack += 1
                        elif formula[j] == ')':
                            stack -= 1
                            if stack == 0:
                                subformula = formula[start_index:j+1]
                                minimized = minimize_func(subformula, **kwargs)
                                formula = formula[:start_index - 1] + " " + "(" + minimized + ")" + formula[j + 1:]
                                index = j
                                break
            index += 1
        return formula
