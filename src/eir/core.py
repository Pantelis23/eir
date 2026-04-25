import abc
import dataclasses
from typing import List, Dict, Union, Any, Optional
import numpy as np

@dataclasses.dataclass(frozen=True)
class Variable:
    name: str

@dataclasses.dataclass(frozen=True)
class DiscreteVar(Variable):
    n: int # Number of states (0..n-1)

@dataclasses.dataclass(frozen=True)
class ContinuousVar(Variable):
    dim: int # Dimension of vector

Assignment = Dict[str, Union[int, np.ndarray, List[float]]]

class Factor(abc.ABC):
    def __init__(self, scope: List[Variable], weight: float = 1.0):
        self.scope = scope
        self.weight = weight

    @abc.abstractmethod
    def eval(self, assign: Assignment) -> float:
        """
        Evaluate UNWEIGHTED energy (lower is better).
        Should handle both hard values (int) and soft values (prob distributions) if possible.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "scope": [v.name for v in self.scope],
            "weight": self.weight
        }

@dataclasses.dataclass
class EIRGraph:
    vars: List[Variable]
    factors: List[Factor]
    
    def __post_init__(self):
        self._var_map = {v.name: v for v in self.vars}

    def get_var(self, name: str) -> Variable:
        return self._var_map[name]

    def energy(self, assign: Assignment) -> float:
        return sum(f.weight * f.eval(assign) for f in self.factors)

    def validate(self, assign: Assignment) -> List[str]:
        errors = []
        for v in self.vars:
            val = assign.get(v.name)
            if val is None:
                errors.append(f"Missing value for {v.name}")
                continue
            
            if isinstance(v, DiscreteVar):
                # Check for hard assignment
                if isinstance(val, (int, np.integer)):
                    if not (0 <= val < v.n):
                        errors.append(f"DiscreteVar {v.name} out of bounds: {val}")
                elif isinstance(val, (list, np.ndarray)):
                    # Soft assignment validation
                    arr = np.array(val)
                    if arr.shape != (v.n,):
                        errors.append(f"DiscreteVar {v.name} soft assignment shape mismatch: {arr.shape} != ({v.n},)")
                    elif not np.all(arr >= 0):
                        errors.append(f"DiscreteVar {v.name} soft assignment contains negative values")
                    elif not np.isclose(np.sum(arr), 1.0, atol=1e-3):
                        errors.append(f"WARN: DiscreteVar {v.name} soft assignment sum {np.sum(arr)} != 1.0")
                else:
                    errors.append(f"DiscreteVar {v.name} invalid type: {type(val)}")
            
            elif isinstance(v, ContinuousVar):
                if isinstance(val, (np.ndarray, list)):
                    arr = np.array(val)
                    if arr.shape != (v.dim,):
                        errors.append(f"ContinuousVar {v.name} shape mismatch: {arr.shape} != ({v.dim},)")
                    if not np.all(np.isfinite(arr)):
                        errors.append(f"ContinuousVar {v.name} contains non-finite values")
                else:
                    errors.append(f"ContinuousVar {v.name} invalid type: {type(val)}")
        return errors