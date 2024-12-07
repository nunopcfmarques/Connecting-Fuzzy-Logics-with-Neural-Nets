from Logic.TLogic import *
from Logic.Parser import *
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

    ast, _ = generate_ast(formula, TLogic.connectives)

    possible_combinations = list(product(values, repeat=num_variables))


    truth_table = {}

    for combination in possible_combinations:
        assignment = {variables[i]: combination[i] for i in range(num_variables)}
        output = evaluate_formula(ast, assignment, TLogic)
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

    os.system("espresso-logic/bin/espresso espresso_file.pla > out.pla")

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
                
                clause += f")∧{binary_vector_to_value[output]})"
                clauses.append(clause)

    final_clause = "V".join(clauses)
    return final_clause




