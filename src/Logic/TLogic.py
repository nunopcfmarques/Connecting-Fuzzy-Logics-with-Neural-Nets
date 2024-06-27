import numpy as np

class Lukasiewicz():
    def __init__(self) -> None:
        self.characters_to_truth_function = {
            "¬": self.neg, 
            "⊙": self.strong_conj, 
            "⊕": self.strong_disj,
            "∧": self.weak_conj, 
            "V": self.weak_disj, 
            "⇒": self.implies, 
            "δ": self.delta,
            "T": self.true,
            "⊥": self.false
        }

        self.connectives = {"¬", "⊙", "⊕", "∧", "V",  "⇒", "δ"}

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
    
    def true(self) -> np.float64:
        return np.float64(1)
    
    def false(self) -> np.float64:
        return np.float64(0)

class Godel():
    def __init__(self) -> None:
        self.characters_to_truth_function = {
            "∧": self.weak_conj, 
            "V": self.weak_disj, 
            "⇒": self.implies, 
            "T": self.true,
            "⊥": self.false
        }

        self.connectives = {"∧", "V", "⇒"}

    #v(A∧B) = min(v(A), v(B))
    def weak_conj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.minimum(x, y))
    
    #v(AVB) = max(v(A), v(B))
    def weak_disj(self, x: np.float64, y: np.float64) -> np.float64:
        return np.float64(np.maximum(x, y))
    
    def implies(self, x: np.float64, y: np.float64) -> np.float64:
        return y if x > y else np.float64(1)
    
    def true(self) -> np.float64:
        return np.float64(1)
    
    def false(self) -> np.float64:
        return np.float64(0)
