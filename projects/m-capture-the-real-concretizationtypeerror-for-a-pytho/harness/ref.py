import numpy as np

class ConcretizationTypeError(TypeError):
    def __init__(self, tracer):
        self.tracer = tracer
        super().__init__(f"Abstract tracer value {tracer} cannot be evaluated in Python boolean context.")

class Tracer:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __bool__(self):
        raise ConcretizationTypeError(self)

    def __repr__(self):
        return f"Tracer({self.name}, shape={self.shape}, dtype={self.dtype})"

def capture_concretization_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ConcretizationTypeError as e:
        return True, e
    except Exception:
        return False, None
    return False, None

class MockJitFunction:
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
    jitted = MockJitFunction(fn)
    for inp in inputs:
        jitted(inp)
    return jitted.retrace_count

def compare_static_vs_array(fn, int_values):
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
