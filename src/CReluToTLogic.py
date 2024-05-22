from src.CReLUNetwork import CReLUNetwork, torch
from src.utils.math_utils import get_lcm
import numpy as np

def sigma_activation(x: float):
    return np.minimum(np.maximum(x, 0), 1)

def is_true(string):
    return string == "1" or string == "(¬0)"

def is_false(string):
    return string == "0" or string == "(¬1)"

# TODO: Type hints
def sigma_construct_rational(w, b, lcm):
    integer_w = lcm * w
    integer_b = lcm * b
    term = sigma_construct(integer_w, integer_b)
    for i in range(1, lcm): # this goes to s - 1 by definition of range in python
        term = f"({term}⊕(δ_{lcm}{sigma_construct(integer_w, integer_b - i)}))"
    return term
    
def sigma_construct(w, b, lcm):
    if np.all(w == 0):
       if b <= 0 or b >= 1:
            return str(int(sigma_activation(b)))
       else:
            term = f"(δ_{lcm})"
            for _ in range(1, int(b * lcm)):
                term = f"({term}⊕(δ_{lcm})"
            return term
        
    elif np.all(w <= 0):
        return f"(¬{sigma_construct(-1*w, -1*b + 1, lcm)})"
    
    elif np.all(w < 1):
        idx = np.where(w > 0)[0][0]
        wl, wr = w.copy(), w.copy()
        wl[idx] = 0
        wr[idx] = 0
        left = sigma_construct(wl, b, lcm)
        right = sigma_construct(wr, b + 1, lcm)
        if is_false(right):
            return '0'
        
        if is_true(left):
            return right
        
        if is_true(right):
            term = f"(δ_{lcm}x{idx + 1})"
            for _ in range(1, int(w[idx] * lcm)): # this goes to s - 1 by definition of range in python
                term = f"({term}⊕(δ_{lcm}x{idx + 1})"

            if is_false(left):
                return term
            
            else:
                return f"({left}⊕{term})"
            
        if is_false(left):
            term = f"(δ_{lcm}x{idx + 1})"
            for _ in range(1, int(w[idx] * lcm)):
                term = f"({term}⊕(δ_{lcm}x{idx + 1})"

            if is_true(right):
                return term
            
            else:
                return f"({term}⊙{right})"
            
        return f"(({left}⊕x{idx+1})⊙{right})"
    
    else:
        idx = np.where(w > 0)[0][0]
        wl, wr = w.copy(), w.copy()
        wl[idx] -= 1
        wr[idx] -= 1
        left = sigma_construct(wl, b, lcm) # results f0
        right = sigma_construct(wr, b + 1, lcm) #results f0 + 1

        if is_false(right): # sigma(f0 ⊕ x) ⊙ 0 = 0 
            return "0"
        
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
        MV_terms[layer] = []
        for neuron in range(CReLU.weights[layer].shape[0]): # number of outputs = number of neurons!
            w = CReLU.weights[layer][neuron].numpy() # gives us the row associated with the neuron
            b = CReLU.biases[layer][neuron].item() # gives us the bias associated with the neuron
            lcm = get_lcm(w + [b]) # TODO we can do this without using lists.
            MV_term = sigma_construct(w, b, lcm)
            print(layer)
            print(MV_term)
            MV_terms[layer].append(MV_term)
