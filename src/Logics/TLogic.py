
class TLogic:

    def __init__(self, L: set) -> None:
        self.language = L
        self.val = []

    def IMPLIES(x, y):
        pass

    def FALSE():
        return 0
    
    def TRUE(self):
        return self.NEG(self.FALSE())

    def NEG(self, x):
        return self.IMPLIES(x, self.FALSE())

    def CONJ(self, x, y):
        return self.IMPLIES(self.NEG(self, x), y)
    
    def DISJ(self, x, y):
        return self.NEG(self.IMPLIES(x, self.NEG(y)))
