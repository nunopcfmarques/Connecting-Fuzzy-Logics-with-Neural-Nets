from Networks.MLP import *
from utils.math_utils import get_lcm

def is_true(string):
    return string == "T" or string == "(¬⊥)"

def is_false(string):
    return string == "⊥" or string == "(¬T)"

def sigma_construct_rational(w, b, lcm):
    integer_w = lcm * w
    integer_b = lcm * b
    term = f'(δ_{lcm} {sigma_construct(integer_w, integer_b)})'
    for i in range(1, lcm): # this goes to s - 1 by definition of range in python
        term = f"({term}⊕(δ_{lcm} {sigma_construct(integer_w, integer_b - i)}))"
    return term

def sigma_construct(w, b):
    if torch.all(w == 0):
        return "T" if int(sigma_activation(b)) == 1 else "⊥"

    elif torch.all(w <= 0):
        return f"(¬{sigma_construct(-1*w, -1*b + 1)})"

    else:
        idx = torch.where(w > 0)[0][0].item()
        wl, wr = w.clone(), w.clone()
        wl[idx] -= 1
        wr[idx] -= 1
        left = sigma_construct(wl, b)
        right = sigma_construct(wr, b + 1)

        if is_false(right): # sigma(f0 ⊕ x) ⊙ 0 = 0 
            return "⊥"

        if is_true(left): # sigma(1 ⊕ x) ⊙ sigma(f0 + 1) = sigma(f0 + 1)
            return right

        if is_true(right): # sigma(f0 ⊕ x) ⊙ 1 == sigma(f0 ⊕ x)
            if is_false(left): # sigma(0 ⊕ x) = x
                return f"x{idx+1}"
            else:
                return f"({left}⊕x{idx+1})"

        if is_false(left): # sigma(0 ⊕ x) ⊙ sigma(f0 + 1) = x ⊙ sigma(f0 + 1)
            if is_true(right):
                return f"x{idx+1}"
            else:
                return f"(x{idx+1}⊙{right})"

        return f"(({left}⊕x{idx+1})⊙{right})"

def construct_MV_terms(CReLU: CReLUNetwork) -> dict:
    MV_terms = {}
    for layer in range(CReLU.num_layers):
        MV_terms[layer] = {}
        for neuron in range(CReLU.weights[layer].shape[0]): # number of outputs = number of neurons!
            w = CReLU.weights[layer][neuron]
            b = CReLU.biases[layer][neuron]

            lcm = get_lcm(w.tolist() + [b.item()])
            if lcm != 1:
                MV_term = sigma_construct_rational(w, b, lcm)
            else:
                MV_term = sigma_construct(w, b)

            MV_terms[layer][neuron + 1] = MV_term
    return MV_terms

def compose_MV_terms(CReLU: CReLUNetwork, MV_terms: dict) -> str:
    for layer in range(1, CReLU.num_layers):
        for neuron in range(CReLU.weights[layer].shape[0]):
            for variable in range(CReLU.weights[layer].shape[1]): # number of inputs
                MV_terms[layer][neuron + 1] = MV_terms[layer][neuron + 1].replace(f'x{variable + 1}', f's{variable + 1}')
        
        for neuron in range(CReLU.weights[layer].shape[0]):
            for variable in range(CReLU.weights[layer].shape[1]):
                MV_terms[layer][neuron + 1] = MV_terms[layer][neuron + 1].replace(f's{variable + 1}', MV_terms[layer - 1][variable + 1])

    return MV_terms[CReLU.num_layers - 1][1]

def tensor_to_assignment(tensor: torch.tensor) -> dict:
    assignment = {}
    for i in range(tensor.size(0)):
        assignment[f'x{i + 1}'] = tensor[i].item()
    return assignment