from src.ReLUNetwork import *
from src.Parser import LanguageParser
import re

class LogicToReLU():
    connective_to_relu = {
        "⊙": ReLUNetwork(
            [np.matrix([1., 1.], dtype=np.float32), np.matrix([1.], dtype=np.float32)],
            [np.array([-1.], dtype=np.float32), np.array([0.], dtype=np.float32)]
        ),
        "¬": ReLUNetwork(
            [np.matrix([1.], dtype=np.float32), np.matrix([-1.], dtype=np.float32)],
            [np.array([0.], dtype=np.float32), np.array([1.], dtype=np.float32)]
        ),
        "⊕": ReLUNetwork(
            [np.matrix([-1., -1.], dtype=np.float32), np.matrix([-1.], dtype=np.float32)],
            [np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32)]
        ),
        "atom":
            ReLUNetwork(
            [np.matrix([1.], dtype=np.float32), np.matrix([1.], dtype=np.float32)],
            [np.array([0.], dtype=np.float32), np.array([0.], dtype=np.float32)]
        ),
    }

    @staticmethod
    def compose_layers_vertically(connectives_list: list[str]) -> ReLUNetwork:
        if re.match('[a-zA-Z]\d*', connectives_list[0]):
            connectives_list[0] = "atom"
        ReLU = LogicToReLU.connective_to_relu[connectives_list[0]]
        for i in range(1, len(connectives_list)):
            if re.match('[a-zA-Z]\d*', connectives_list[i]):
                connectives_list[i] = "atom"
            ReLU = ReLUNetworkOperations.compose_vertically(ReLU, LogicToReLU.connective_to_relu[connectives_list[i]])
        return ReLU


    @staticmethod
    def formula_to_ReLU(formula: str) -> ReLUNetwork:
        formula_tree, inputs, depth = LanguageParser.formula_to_tree(formula)
        ReLU = LogicToReLU.compose_layers_vertically(formula_tree[0])
        for layer in range(0, depth):
            ReLU1 = LogicToReLU.compose_layers_vertically(formula_tree[layer + 1])
            ReLU = ReLUNetworkOperations.compose_horizontally(ReLU1, ReLU)
        return ReLU, inputs
    
    @staticmethod
    def valuation_to_tensor(val: dict, inputs):
        val_inputs = []
        for input in inputs:
            val_inputs.append(val[input[0]])
        
        return torch.tensor(val_inputs, dtype=torch.float32)
