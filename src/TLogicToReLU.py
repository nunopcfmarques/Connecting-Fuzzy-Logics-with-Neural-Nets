from src.ReLUNetwork import *
from src.TLogics import *
from copy import deepcopy

'''
This class will have two main fields:
connectives_to_ReLU: A dict whose keys are connectives and whose values are the corresponding ReLU Networks for those values
TLogic: The logic that gives semantic meaning to the formulas we will be translating.
'''
class LogicToRelu():
    def __init__(self, connectives_to_ReLU: dict[str, ReLUNetwork], Logic: TLogic) -> None:
        self.connectives_to_ReLU = connectives_to_ReLU
        self.TLogic = Logic

    def ast_to_ReLU(self, root: Tree.Node, max_depth: int) -> ReLUNetwork:
        ReLU = ReLUNetwork()
        ReLU_v = ReLUNetwork()


        #Aqui isto está a criar para uma rede de 2 Layers para um átomo o que é um bocado parvo
        expand_queue = [root]

        connectives = set(self.connectives_to_ReLU.keys())

        while expand_queue:

            node = expand_queue.pop(0)
            connective = node.data[0] if node.data[0] in connectives else ""
            
            if connective == "" and node.depth < max_depth - 1:
                children = Tree.Node(node.data, node.depth + 1)
                expand_queue.append(children)
            else:
                for children in Tree.get_children(node):
                    if children.data[0] in connectives or (children.depth < max_depth and children.data[0] not in connectives):
                        expand_queue.append(children)
            
            if ReLU_v.weights:
                if connective == "δ":
                    ReLU_v.vertically_append_ReLUs(self.connectives_to_ReLU[connective](node.data[1:]))
                else:
                    ReLU_v.vertically_append_ReLUs(self.connectives_to_ReLU[connective])
            else:
                if connective == "δ":
                    ReLU_v = deepcopy(self.connectives_to_ReLU[connective](node.data[1:]))
                else:
                    ReLU_v = deepcopy(self.connectives_to_ReLU[connective])
            
            if not expand_queue or node.depth != expand_queue[0].depth:
                if ReLU.weights:
                    ReLU.horizontally_append_ReLUs(ReLU_v)
                else:
                    if connective == "δ":
                        ReLU = deepcopy(self.connectives_to_ReLU[connective](node.data[1:]))
                    else:
                        ReLU = deepcopy(self.connectives_to_ReLU[connective])
                ReLU_v = ReLUNetwork()

        return ReLU
    
    def calculate_maximum_depth(self, formula: str) -> int:
        if len(formula) == 1:
            return 0
        else:
            lformula, rformula, connective = self.TLogic.subdivide_formula(formula)
            num_layers = self.connectives_to_ReLU[connective].num_layers
            if connective == "¬":
                return num_layers + self.calculate_maximum_depth(lformula)
            else:
                return num_layers + np.maximum(self.calculate_maximum_depth(lformula), self.calculate_maximum_depth(rformula))
    
    @staticmethod
    def valuation_to_tensor(val: dict, formula: str) -> torch.Tensor:
        return torch.tensor([val[char] for char in formula if char in val], dtype=torch.float64)


