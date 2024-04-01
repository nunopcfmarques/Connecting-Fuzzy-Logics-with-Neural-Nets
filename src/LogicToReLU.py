from src.ReLUNetwork import *
from src.TLogics import *
from copy import deepcopy

connective_to_ReLU = {
    "⊙": ReLUNetwork(
        [np.matrix([1., 1.], dtype=np.float64), np.matrix([1.], dtype=np.float64)],
        [np.array([-1.], dtype=np.float64), np.array([0.], dtype=np.float64)]
    ),
    "¬": ReLUNetwork(
        [np.matrix([1.], dtype=np.float64), np.matrix([-1.], dtype=np.float64)],
        [np.array([0.], dtype=np.float64), np.array([1.], dtype=np.float64)]
    ),
    "⊕": ReLUNetwork(
        [np.matrix([-1., -1.], dtype=np.float64), np.matrix([-1.], dtype=np.float64)],
        [np.array([1.], dtype=np.float64), np.array([1.], dtype=np.float64)]
    ),
    "": ReLUNetwork(
        [np.matrix([1.], dtype=np.float64), np.matrix([1.], dtype=np.float64)],
        [np.array([0.], dtype=np.float64), np.array([0.], dtype=np.float64)]
    ),
}

def ast_to_ReLU(root: Tree.Node, max_depth: int) -> ReLUNetwork:
    ReLU = ReLUNetwork()
    ReLU_v = ReLUNetwork()

    expand_queue = [root]

    while expand_queue:
        node = expand_queue.pop(0)
        connective = node.data if node.data in TLogic.connectives else ""
        
        if node.data not in TLogic.connectives and node.depth < max_depth - 1:
            children = Tree.Node(node.data, node.depth + 1)
            expand_queue.append(children)
        else:
            for children in Tree.get_children(node):
                if children.data in TLogic.connectives or (children.depth < max_depth and children.data not in TLogic.connectives):
                    expand_queue.append(children)
        
        if ReLU_v.weights:
            ReLU_v.compose_vertically(connective_to_ReLU[connective])
        else:
            ReLU_v = deepcopy(connective_to_ReLU[connective])
        
        if not expand_queue or node.depth != expand_queue[0].depth:
            if ReLU.weights:
                ReLU.compose_horizontally(ReLU_v)
            else:
                ReLU = deepcopy(connective_to_ReLU[connective])
            ReLU_v = ReLUNetwork()

    return ReLU

def valuation_to_tensor(val: dict, formula: str):
    return torch.tensor([val[char] for char in formula if char.isalpha()], dtype=torch.float64)


