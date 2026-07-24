import numpy as np

def _ref(trace, new_inputs):
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

def grade(sol, fx) -> dict:
    trace = [
        (np.array([[1, 2], [3, 4]]), "input1"),
        (np.array([[5, 6], [7, 8]]), "input2")
    ]
    new_inputs = [
        np.array([[1, 2], [3, 4]]),  # No change
        np.array([[1, 2, 3]]),        # Shape change
        np.array([[1.0, 2.0], [3.0, 4.0]])  # Dtype change
    ]
    
    expected = _ref(trace, new_inputs)
    try:
        got = sol.generate_recompile_guards(trace, new_inputs)
    except Exception:
        return {"exact_match": 0.0}
    
    return {"exact_match": float(got == expected)}
