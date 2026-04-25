from typing import Protocol, Any, Tuple, Dict
from ..core import EIRGraph

class Adapter(Protocol):
    def build_graph(self, context: Any) -> Tuple[EIRGraph, Dict[str, Any], str]:
        """
        Converts external context (problem state) into an EIR Graph.
        Returns:
            - graph: The optimization problem
            - initial_assignment: Hint/Start state
            - action_var_name: The name of the variable representing the primary decision/action
        """
        ...
        
    def decode_action(self, solution: Dict[str, Any]) -> Any:
        """
        Converts the raw dictionary solution back to the domain-specific action object.
        """
        ...

class Planner:
    def __init__(self, adapter: Adapter, backend_name: str = "field"):
        self.adapter = adapter
        self.backend_name = backend_name
        
    def solve(self, context: Any) -> Any:
        # 1. Adapt
        graph, init_assign, action_var = self.adapter.build_graph(context)
        
        # 2. Solve
        if self.backend_name == "field":
            from ..backends.field import FieldBackend
            # TODO: Pass init_assign if supported
            solver = FieldBackend()
            result = solver.solve(graph)
            raw_solution = result.assignment
            
        elif self.backend_name == "verify":
            from ..backends.verify_cpsat import VerifyCPSATBackend
            solver = VerifyCPSATBackend()
            result = solver.solve(graph)
            if result.assignment is None:
                raise RuntimeError(f"Planner failed: {result.status}")
            raw_solution = result.assignment
            
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
            
        # 3. Decode
        return self.adapter.decode_action(raw_solution)
