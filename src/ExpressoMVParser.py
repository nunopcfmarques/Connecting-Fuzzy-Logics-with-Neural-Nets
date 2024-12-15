from src.TLogics import *
from itertools import product
import os

def BinaryVectorEncoder(values: list) -> tuple[dict, dict]:

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

def FormulaToTruthTable(formula: str, TLogic, variables: list[str], values:list, value_to_binary_vector: dict) -> dict:

    num_variables = len(variables)

    ast, _ = TLogic.generate_ast(formula)

    possible_combinations = list(product(values, repeat=num_variables))


    truth_table = {}

    for combination in possible_combinations:
        assignment = {variables[i]: combination[i] for i in range(num_variables)}
        output = TLogic.evaluate_formula(ast, assignment)
        truth_table[''.join([value_to_binary_vector[combination[i]] for i in range(num_variables)])] = value_to_binary_vector[output]


    return truth_table

def ToExpressoMVParser(formula: str, TLogic, variables: list[str], values:list, value_to_binary_vector: dict):

    num_variables = len(variables)

    value_to_binary_vector, _ = BinaryVectorEncoder(values)

    size_of_variable = len(next(iter(value_to_binary_vector.values())))

    truth_table = FormulaToTruthTable(formula, TLogic, variables, values, value_to_binary_vector)

    with open('espresso_file.pla', 'w+') as espresso_file:
        espresso_file.write(f'.i {num_variables * size_of_variable}\n')
        espresso_file.write(f'.o {size_of_variable}\n') # + 1 because of output
        for inputs in truth_table.keys():
            espresso_file.write(f'{inputs} {truth_table[inputs]}\n')

    os.system("src/espresso-logic/bin/espresso espresso_file.pla > out.pla")

    return

def FromExpressoMVParser(binary_vector_to_value: dict) -> str:
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
                            condition = "∧".join(f"(x{index+1}!={val})" for val in unmatching_values)
                            input_conditions.append(condition)
                        else:
                            condition = "∨".join(f"(x{index+1}=={val})" for val in matching_values)
                            input_conditions.append(f"({condition})")
                    else:
                        input_conditions.append(f"(x{index+1}=={binary_vector_to_value[inp]})")
                
                clause += "∧".join(input_conditions)
                
                clause += f"∧{binary_vector_to_value[output]})"
                clauses.append(clause)

    final_clause = "V".join(clauses)
    print(final_clause)
    return final_clause

def minimize_lukasiewicz(formula: str, values: list) -> str:
    variables = re.findall(r'\b[a-zA-Z]+\d*\b', formula)
    
    # Remove duplicates
    unique_variables = list(set(variables))
    
    # Sort by the number part of the variable
    sorted_variables = sorted(unique_variables, key=lambda v: (re.match(r'[a-zA-Z]+', v).group(0), int(re.search(r'\d+', v).group(0)) if re.search(r'\d+', v) else float('inf')))

    value_to_binary_vector, binary_vector_to_value = BinaryVectorEncoder(values)
    ToExpressoMVParser(formula, Lukasiewicz(), sorted_variables, values, value_to_binary_vector)
    normal_formula = FromExpressoMVParser(binary_vector_to_value)

    return normal_formula



def minimize_delta_formulas(formula: str, values: list) -> str:
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
                            print(subformula)
                            minimized = minimize_lukasiewicz(subformula, values)
                            print(minimized)
                            formula = formula[:start_index - 1] + " " + "(" + minimized + ")" + formula[j + 1:]
                            index = j
                            break
        index += 1
    return formula
                

expression = "(¬((δ_2 (((x3⊙(¬x2))⊕x1)⊙((¬x2)⊕x3)))⊕(δ_2 (x1⊙(x3⊙(¬x2))))))"

formula = minimize_delta_formulas(expression, [0, 0.5, 1])

print(formula)

TLogic = Lukasiewicz()
ast, _ = TLogic.generate_ast(formula)
print(TLogic.evaluate_formula(ast, {"x1":0 , "x2": 0, "x3": 0}))