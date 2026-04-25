import json
import warnings
import numpy as np
from typing import Any
from .core import EIRGraph, DiscreteVar, ContinuousVar
from .factors import logic, quadratic, onehot

def default_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def serialize_graph(graph: EIRGraph) -> str:
    data = {
        "vars": [],
        "factors": [f.to_dict() for f in graph.factors]
    }
    
    for v in graph.vars:
        if isinstance(v, DiscreteVar):
            data["vars"].append({"type": "DiscreteVar", "name": v.name, "n": v.n})
        elif isinstance(v, ContinuousVar):
            data["vars"].append({"type": "ContinuousVar", "name": v.name, "dim": v.dim})
            
    return json.dumps(data, default=default_encoder, indent=2)

def deserialize_graph(json_str: str, allow_unknown: bool = False) -> EIRGraph:
    data = json.loads(json_str)
    vars_list = []
    var_map = {}
    
    for v_data in data["vars"]:
        if v_data["type"] == "DiscreteVar":
            v = DiscreteVar(v_data["name"], v_data["n"])
        elif v_data["type"] == "ContinuousVar":
            v = ContinuousVar(v_data["name"], v_data["dim"])
        else:
            raise ValueError(f"Unknown var type: {v_data['type']}")
        vars_list.append(v)
        var_map[v.name] = v
        
    factors = []
    from .factors import gated, bias, dynamics
    
    # Registry of available factor classes
    factor_classes = {
        "QuadraticFactor": quadratic.QuadraticFactor,
        "OneHotFactor": onehot.OneHotFactor,
        "ImplicationFactor": logic.ImplicationFactor,
        "XorFactor": logic.XorFactor,
        "AndFactor": logic.AndFactor,
        "GatedTargetsFactor": gated.GatedTargetsFactor,
        "ActionBiasFactor": bias.ActionBiasFactor,
        "DynamicsConsistencyFactor": dynamics.DynamicsConsistencyFactor,
        "RewardFactor": dynamics.RewardFactor
    }
    
    for f_data in data["factors"]:
        cls_name = f_data["type"]
        if cls_name not in factor_classes:
            msg = f"Unknown factor type: {cls_name}. Supported: {list(factor_classes.keys())}"
            if allow_unknown:
                warnings.warn(msg)
                continue
            else:
                raise ValueError(msg)
            
        cls = factor_classes[cls_name]
        scope = [var_map[n] for n in f_data["scope"]]
        
        # Extract extra kwargs (excluding type, scope, weight)
        kwargs = {k: v for k, v in f_data.items() if k not in ["type", "scope", "weight"]}
        
        factors.append(cls(scope=scope, weight=f_data.get("weight", 1.0), **kwargs))
        
    return EIRGraph(vars_list, factors)