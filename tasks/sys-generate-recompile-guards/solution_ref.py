from typing import List, Tuple
import numpy as np

def generate_recompile_guards(trace: List[Tuple[np.ndarray, str]], new_inputs: List[np.ndarray]) -> List[bool]:
    last_input = trace[-1][0]
    last_shape = last_input.shape
    last_dtype = last_input.dtype
    last_rank = len(last_shape)

    results = []
    for new_input in new_inputs:
        new_shape = new_input.shape
        new_dtype = new_input.dtype
        new_rank = len(new_shape)

        recompile = (last_shape != new_shape) or (last_dtype != new_dtype) or (last_rank != new_rank)
        results.append(recompile)
    
    return results
