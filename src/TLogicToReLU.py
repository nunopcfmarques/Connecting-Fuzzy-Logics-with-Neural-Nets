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

    def add_or_change_connective_To_ReLU(self, connective: str, ReLU: ReLUNetwork) -> 'LogicToRelu':
        self.connectives_to_ReLU[connective] = ReLU
        return self

    def ast_to_ReLU(self, root: Tree.Node, max_depth: int) -> ReLUNetwork:
        ReLU = ReLUNetwork()
        ReLU_v = ReLUNetwork()

        #Aqui isto está a criar para uma rede de 2 Layers para um átomo o que é um bocado parvo
        expand_queue = [root]

        connectives = set(self.connectives_to_ReLU.keys())

        while expand_queue:
            node = expand_queue.pop(0)
            connective = node.data if node.data in connectives else ""
            
            if node.data not in connectives and node.depth < max_depth - 1:
                children = Tree.Node(node.data, node.depth + 1)
                expand_queue.append(children)
            else:
                for children in Tree.get_children(node):
                    if children.data in connectives or (children.depth < max_depth and children.data not in connectives):
                        expand_queue.append(children)
            
            if ReLU_v.weights:
                ReLU_v.vertically_append_ReLUs(self.connectives_to_ReLU[connective])
            else:
                ReLU_v = deepcopy(self.connectives_to_ReLU[connective])
            
            if not expand_queue or node.depth != expand_queue[0].depth:
                if ReLU.weights:
                    ReLU.horizontally_append_ReLUs(ReLU_v)
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
    
    #TODO
    '''
    def construct_ReLU_for_connective(self, connective: str) -> ReLUNetwork:
        uses the uderlying logic and constructs the ReLU given the relation of the new connective to already built connectives
    '''
    
    @staticmethod
    def valuation_to_tensor(val: dict, formula: str) -> torch.tensor:
        return torch.tensor([val[char] for char in formula if char.isalpha()], dtype=torch.float64)


