from TLogic import *

class Lukasiewicz(TLogic):

    def IMPLIES(self, x, y):
        return min(1, 1 - x + y)

    def CONJ(self, x, y):
        return min(1, x + y)

    def DISJ(self, x, y):
        return max(0, x + y - 1)

    def NEG(self, x):
        return 1 - x