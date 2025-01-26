import numpy as np
from src.Logic.Connective import Connective

class Lukasiewicz():
    def __init__(self) -> None:
        # initialize connectives
        self.connectives = {
            "¬": Connective("¬", 1, self.neg),
            "⊙": Connective("⊙", 2, self.strong_conj),
            "⊕": Connective("⊕", 2, self.strong_disj),
            "∧": Connective("∧", 2, self.weak_conj),
            "V": Connective("V", 2, self.weak_disj),
            "⇒": Connective("⇒", 2, self.implies),
            "δ": Connective("δ", 1, self.delta),
            "=": Connective("=", 2, self.equals),
            "!": Connective("!", 2, self.unequals)
        }

    #v(A → B) = min(1,1−v(A) +v(B))
    def implies(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(1, 1 - x + y))

    #v(A⊙B) = max(0,v(A) +v(B)−1)
    def strong_conj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.maximum(0, x + y - 1))

    #v(A⊕B) = min(1,v(A) +v(B))
    def strong_disj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(1, x + y))
    
    #v(A∧B) = min(v(A), v(B))
    def weak_conj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(x, y))
    
    #v(AVB) = max(v(A), v(B))
    def weak_disj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.maximum(x, y))
    
    #v(¬A) = 1−v(A)
    def neg(self, x: np.float64) -> np.float64:
        return np.float64(1 - x)
    
    def delta(self, i: int, x: np.float64 = 1.) -> np.float64:
        return np.float64(x / int(i))
    
    def equals(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(x == y)
    
    def unequals(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(x != y)


class Godel():
    def __init__(self) -> None:
        self.connectives = {
            "∧": Connective("∧", 2, self.weak_conj),
            "V": Connective("V", 2, self.weak_disj),
            "⇒": Connective("⇒", 2, self.implies),
        }

    #v(A∧B) = min(v(A), v(B))
    def weak_conj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(x, y))
    
    #v(AVB) = max(v(A), v(B))
    def weak_disj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.maximum(x, y))
    
    def implies(self, x: np.float64, y: np.float64) -> np.float64:
        return y if x > y else np.float64(1)