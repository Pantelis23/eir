import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ..core import EIRGraph, DiscreteVar, ContinuousVar

@dataclass
class VerifyResult:
    status: str
    assignment: Optional[Dict[str, Any]]
    objective_value: Optional[float] = None
    unsupported_factors: Optional[List[str]] = None
    
class VerifyCPSATBackend:
    def __init__(self, strict: bool = True):
        self.strict = strict
        try:
            from ortools.sat.python import cp_model
            self.model_cls = cp_model.CpModel
            self.solver_cls = cp_model.CpSolver
            self.status_map = {
                cp_model.OPTIMAL: "OPTIMAL",
                cp_model.FEASIBLE: "FEASIBLE",
                cp_model.INFEASIBLE: "INFEASIBLE",
                cp_model.MODEL_INVALID: "INVALID",
                cp_model.UNKNOWN: "UNKNOWN"
            }
        except ImportError:
            raise ImportError("ortools is required for VerifyCPSATBackend. Install with 'pip install ortools'")

    def solve(self, graph: EIRGraph) -> VerifyResult:
        model = self.model_cls()
        varmap = {}
        unsupported = []
        
        # 1. Create variables
        for v in graph.vars:
            if isinstance(v, DiscreteVar):
                varmap[v.name] = model.NewIntVar(0, v.n - 1, v.name)
            elif isinstance(v, ContinuousVar):
                # CP-SAT does not support continuous variables.
                print(f"Warning: Continuous variable {v.name} ignored by CP-SAT backend.")
            
        # 2. Add factors
        for f in graph.factors:
            if hasattr(f, 'compile_cpsat'):
                try:
                    f.compile_cpsat(model, varmap)
                except Exception as e:
                    # In strict mode, we might want to catch this too if it's a structural issue,
                    # but typically KeyError means variable missing (e.g. continuous).
                    # If the factor claims to support CP-SAT but fails, that's an error.
                    msg = f"Factor {f} compilation failed: {e}"
                    if self.strict:
                        raise RuntimeError(msg) from e
                    print(msg)
                    unsupported.append(str(f))
            else:
                msg = f"Factor {type(f).__name__} does not support CP-SAT compilation."
                if self.strict:
                    raise NotImplementedError(msg)
                print(msg)
                unsupported.append(type(f).__name__)

        # 3. Solve
        solver = self.solver_cls()
        status = solver.Solve(model)
        
        res_status = self.status_map.get(status, "UNKNOWN")
        assignment = None
        obj_val = 0.0 # CP-SAT usually integral objective but we have no objective here yet
        
        if status == 4 or status == 2: # OPTIMAL or FEASIBLE
            assignment = {}
            for v_name, cp_var in varmap.items():
                assignment[v_name] = solver.Value(cp_var)
            obj_val = solver.ObjectiveValue()
            
        return VerifyResult(
            status=res_status, 
            assignment=assignment, 
            objective_value=obj_val,
            unsupported_factors=unsupported if unsupported else None
        )