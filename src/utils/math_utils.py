from fractions import Fraction
import math

def get_lcm(coefficients: list[float]) -> int:
    return math.lcm(*[Fraction(coefficient).limit_denominator().denominator for coefficient in coefficients])
