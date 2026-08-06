class ConcretizationTypeError(TypeError):
    """Raised when an abstract tracer is evaluated in a Python boolean context."""
    def __init__(self, tracer):
        self.tracer = tracer
        super().__init__(f"Abstract tracer value {tracer} cannot be evaluated in Python boolean context.")

class Tracer:
    """Represents a traced symbolic value."""
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __bool__(self):
        raise ConcretizationTypeError(self)

    def __repr__(self):
        return f"Tracer({self.name}, shape={self.shape}, dtype={self.dtype})"

def capture_concretization_error(fn, *args, **kwargs):
    """Executes fn(*args, **kwargs) and returns (True, error) if ConcretizationTypeError occurs."""
    try:
        fn(*args, **kwargs)
    except ConcretizationTypeError as e:
        return True, e
    except Exception:
        return False, None
    return False, None
