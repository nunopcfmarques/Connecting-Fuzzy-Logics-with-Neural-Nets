from src.Logics.TLogic import *

class Lukasiewicz(TLogic):

    @staticmethod
    def IMPLIES(x, y):
        return min(1, 1 - x + y)

    @staticmethod
    def CONJ(x, y):
        return max(0, x + y - 1)

    @staticmethod
    def DISJ(x, y):
        return min(1, x + y)

    @staticmethod
    def NEG(x):
        return 1 - x