import numpy as np
from jaxcomp.tracer import Tracer, ConcretizationTypeError

class MockJitFunction:
    """Wraps a function and tracks retracing events based on input signature changes."""
    def __init__(self, fn, static_argnums=()):
        self.fn = fn
        self.static_argnums = tuple(static_argnums) if isinstance(static_argnums, (list, tuple)) else (static_argnums,)
        self.compilations = {}
        self.retrace_count = 0

    def __call__(self, *args):
        sig = []
        for i, arg in enumerate(args):
            if i in self.static_argnums:
                sig.append(("static", arg))
            elif hasattr(arg, "shape") and hasattr(arg, "dtype"):
                sig.append(("array", getattr(arg, "shape"), getattr(arg, "dtype")))
            else:
                sig.append(("val", type(arg), arg))
        key = tuple(sig)
        if key not in self.compilations:
            self.retrace_count += 1
            traced_args = []
            for i, arg in enumerate(args):
                if i in self.static_argnums:
                    traced_args.append(arg)
                elif hasattr(arg, "shape") and hasattr(arg, "dtype"):
                    traced_args.append(Tracer(f"arg_{i}", getattr(arg, "shape"), getattr(arg, "dtype")))
                else:
                    traced_args.append(arg)
            try:
                self.fn(*traced_args)
            except ConcretizationTypeError:
                pass
            self.compilations[key] = True

        if hasattr(args[0], "shape"):
            return np.ones(args[0].shape)
        return 1

def trace_and_count_retraces(fn, inputs):
    """Executes fn over inputs and returns total retraces performed."""
    jitted = MockJitFunction(fn)
    for inp in inputs:
        jitted(inp)
    return jitted.retrace_count
