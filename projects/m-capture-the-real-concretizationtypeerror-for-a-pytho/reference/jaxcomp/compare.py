import numpy as np
from jaxcomp.retrace import MockJitFunction

def compare_static_vs_array(fn, int_values):
    """Compares retrace count behavior when using static_argnums vs array arguments."""
    jit_static = MockJitFunction(fn, static_argnums=(0,))
    jit_array = MockJitFunction(fn)

    for val in int_values:
        jit_static(val)
        arr = np.array([val], dtype=np.int32)
        jit_array(arr)

    return {
        "static_retrace_count": jit_static.retrace_count,
        "array_retrace_count": jit_array.retrace_count
    }
