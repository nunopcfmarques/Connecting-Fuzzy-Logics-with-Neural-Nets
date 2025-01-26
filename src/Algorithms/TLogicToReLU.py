from src.Networks.MLP import *
from src.Logic.Parser import *
from copy import deepcopy

def construct_lukasiewicz_delta_ReLU(i: int):
    x = np.float64(1 / int(i))
    return ReLUNetwork(
        [torch.tensor([[x]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    )

Lukasiewicz_connectives_to_ReLU = {
    "¬": ReLUNetwork(
        [torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[-1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
    ),
    "⊙": ReLUNetwork(
        [torch.tensor([[1., 1.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([-1.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "⊕": ReLUNetwork(
        [torch.tensor([[-1., -1.]], dtype=torch.float64), torch.tensor([[-1.]], dtype=torch.float64)],
        [torch.tensor([1.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
    ),
    "∧": ReLUNetwork(
        [torch.tensor([[1., 0.], [1., -1.]], dtype=torch.float64), torch.tensor([[1., -1.]], dtype=torch.float64)],
        [torch.tensor([0., 0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "V": ReLUNetwork(
        [torch.tensor([[1., 0.], [-1., 1.]], dtype=torch.float64), torch.tensor([[1., 1.]], dtype=torch.float64)],
        [torch.tensor([0., 0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "∧": ReLUNetwork(
        [torch.tensor([[1., 0.], [1., -1.]], dtype=torch.float64), torch.tensor([[1., -1.]], dtype=torch.float64)],
        [torch.tensor([0., 0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "⇒": ReLUNetwork(
        [torch.tensor([[1., -1.]], dtype=torch.float64), torch.tensor([[-1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
    ),
    "i": ReLUNetwork(
        [torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "δ": construct_lukasiewicz_delta_ReLU
}

Godel_connectives_to_ReLU = {
    "∧": ReLUNetwork(
        [torch.tensor([[1., 0.], [1., -1.]], dtype=torch.float64), torch.tensor([[1., -1.]], dtype=torch.float64)],
        [torch.tensor([0., 0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "V": ReLUNetwork(
        [torch.tensor([[1., 0.], [-1., 1.]], dtype=torch.float64), torch.tensor([[1., 1.]], dtype=torch.float64)],
        [torch.tensor([0., 0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
    "i": ReLUNetwork(
        [torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)],
        [torch.tensor([0.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64)]
    ),
}


def ast_to_ReLU(root: Tree.Node, max_depth: int, connectives_to_ReLU: dict) -> ReLUNetwork:
    ReLU = ReLUNetwork()
    ReLU_v = ReLUNetwork()

    expand_queue = [root]

    connectives = set(connectives_to_ReLU)

    while expand_queue:

        node = expand_queue.pop(0)
        connective = node.data[0] if node.data[0] in connectives else "i"
            
        if connective == "i" and node.depth < max_depth - 1:
            children = Tree.Node(node.data, node.depth + 1)
            expand_queue.append(children)
        else:
            for children in Tree.get_children(node):
                if children.data[0] in connectives or (children.depth < max_depth and children.data[0] not in connectives):
                    expand_queue.append(children)
            
        if ReLU_v.weights:
            if connective == "δ":
                ReLU_v.vertically_append_MLPs(connectives_to_ReLU[connective](node.data[1:]))
            else:
                ReLU_v.vertically_append_MLPs(connectives_to_ReLU[connective])
        else:
            if connective == "δ":
                    ReLU_v = deepcopy(connectives_to_ReLU[connective](node.data[1:]))
            else:
                ReLU_v = deepcopy(connectives_to_ReLU[connective])
            
        if not expand_queue or node.depth != expand_queue[0].depth:
            if ReLU.weights:
                    ReLU.horizontally_append_MLPs(ReLU_v)
            else:
                if connective == "δ":
                    ReLU = deepcopy(connectives_to_ReLU[connective](node.data[1:]))
                else:
                    ReLU = deepcopy(connectives_to_ReLU[connective])
            ReLU_v = ReLUNetwork()

    return ReLU

def assignment_to_tensor(assignment: dict, formula: str) -> torch.Tensor:
    return torch.tensor([assignment[char] for char in formula if char in assignment], dtype=torch.float64)