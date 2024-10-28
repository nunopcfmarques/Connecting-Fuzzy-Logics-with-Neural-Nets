from pyscipopt import *
from src.TLogics import *

def ParseToSCIOPT(m: Model, formula: str, atom_vars={}):
    if TLogic.is_atom(formula):
        if formula not in atom_vars:
            atom_vars[formula] = m.addVar(vtype="C", lb=0, ub=1, name=formula)
        return atom_vars[formula]

    elif TLogic.is_constant(formula):
        return float(formula)

    else:
        l_formula, r_formula, connective = TLogic.subdivide_formula(formula)

        if connective == "¬":
            left_expr = ParseToSCIOPT(m, l_formula, atom_vars)
                
            neg_var = m.addVar(vtype="C", lb=0, ub=1, name=f"neg({l_formula})")
            m.addCons(neg_var == 1 - left_expr)
            return neg_var
                
        elif connective == "⊙":
            left_expr = ParseToSCIOPT(m, l_formula, atom_vars)
            right_expr = ParseToSCIOPT(m, r_formula, atom_vars)

            conj_var = m.addVar(vtype="C", lb=0, ub=1, name=f"conj({l_formula},{r_formula})")
            m.addCons(conj_var <= left_expr)
            m.addCons(conj_var <= right_expr)
            m.addCons(conj_var >= left_expr + right_expr - 1)
            return conj_var

        elif connective == "⊕":
            left_expr = ParseToSCIOPT(m, l_formula, atom_vars)
            right_expr = ParseToSCIOPT(m, r_formula, atom_vars)

            disj_var = m.addVar(vtype="C", lb=0, ub=1, name=f"disj({l_formula},{r_formula})")
            m.addCons(disj_var >= left_expr)
            m.addCons(disj_var >= right_expr)
            m.addCons(disj_var <= left_expr + right_expr)
            m.addCons(disj_var <= 1)
            return disj_var
        
        elif connective == "⇒":
            left_expr = ParseToSCIOPT(m, l_formula, atom_vars)
            right_expr = ParseToSCIOPT(m, r_formula, atom_vars)

            impl_var = m.addVar(vtype="C", lb=0, ub=1, name=f"impl({l_formula},{r_formula})")
            m.addCons(impl_var >= right_expr)
            m.addCons(impl_var >= 1 - left_expr)
            m.addCons(impl_var <= 1 - left_expr + right_expr)
            return impl_var

        elif connective[0] == "δ":
            left_expr = ParseToSCIOPT(m, l_formula, atom_vars)
            n = int(connective[1:])

            delta_var = m.addVar(vtype="C", lb=0, ub=1, name=f"delta_{n}({l_formula})")
            m.addCons(delta_var == left_expr / n)
            return delta_var

        return None

def SolveFormulaMILP(formula: str):
    model = Model()
    model.hideOutput()
    
    atom_vars = {}
    final_expr = ParseToSCIOPT(model, formula, atom_vars)
    
    if final_expr is not None:
        model.addCons(final_expr == 1)
        model.optimize()

        for var in model.getVars():
            var_value = model.getVal(var)
            print(f"{var.name}: {var_value}")