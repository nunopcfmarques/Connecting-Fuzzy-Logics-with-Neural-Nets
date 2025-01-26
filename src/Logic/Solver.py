from z3 import *
from src.Logic.Parser import *

# Apparently we need to define Lukasiewicz connectives to be interpreted by Z3

#v(A → B) = min(1,1−v(A) +v(B))
def IMPLIES(x: ArithRef, y: ArithRef) -> ArithRef:
        return If(1 - x + y < 1, 1 - x + y, 1)

#v(A⊙B) = max(0,v(A) +v(B)−1)    
def CONJ(x: ArithRef, y: ArithRef) -> ArithRef:
        return If(x + y - 1 > 0, x + y - 1, 0)

 #v(A⊕B) = min(1,v(A) +v(B))
def DISJ(x: ArithRef, y: ArithRef) -> ArithRef:
    return If(x + y < 1, x + y, 1)
    
#v(¬A) = 1−v(A)
def NEG(x: ArithRef) -> ArithRef:
    return 1 - x
    
def DELTA(i: ArithRef, x: ArithRef) -> ArithRef:
    return x / i

    
def ParseToZ3(s: Solver, formula: str, Parser, atoms = set()) -> str:
    if Parser.is_atom(formula):
        if formula not in atoms:
            atoms.add(formula)
            s.add(Real(formula) >= 0)
            s.add(Real(formula) <= 1)
        return f"Real('{formula}')"
        
    elif Parser.is_constant(formula):
        if formula == "1":
            return "RealVal(1)"
        else:
            return "RealVal(0)" 

    else:
        l_formula, r_formula, connective = Parser.subdivide_formula(formula)
        
        if connective == "¬":
            left_expr = ParseToZ3(s, l_formula, atoms)
            return f"NEG({left_expr})"
                
        elif connective[0] == "δ":
            left_expr = ParseToZ3(s, l_formula, atoms)
            return f"DELTA({connective[1:]}, {left_expr})"

        else:
            left_expr = ParseToZ3(s, l_formula, atoms)
            right_expr = ParseToZ3(s, r_formula, atoms)
            
            if connective == "⇒":
                return f"IMPLIES({left_expr}, {right_expr})"
                
            elif connective == "⊙":
                return f"CONJ({left_expr}, {right_expr})"
                
            elif connective == "⊕":
                return f"DISJ({left_expr}, {right_expr})"

        return ""

def SolveFormulaSMT(formula: str, target) -> tuple[bool, ModelRef | None]:

    s = SolverFor("LRA")

    parsed_formula = ParseToZ3(s, formula, Parser())

    s.add(eval(parsed_formula) == target)

    if s.check() == sat:
        return True, s.model()
    else:
        return False, None