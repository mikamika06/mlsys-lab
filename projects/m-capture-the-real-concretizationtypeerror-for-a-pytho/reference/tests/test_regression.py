import numpy as np
from jaxcomp.tracer import Tracer, capture_concretization_error, ConcretizationTypeError
from jaxcomp.retrace import trace_and_count_retraces

def test_concretization_error_handling():
    def dummy(x):
        if x:
            return 1
        return 0

    t = Tracer("x", (2, 2), np.float32)
    caught, err = capture_concretization_error(dummy, t)
    assert caught is True
    assert isinstance(err, ConcretizationTypeError)
    assert err.tracer is t

    def other_fn(x):
        raise ValueError("other error")

    caught_other, err_other = capture_concretization_error(other_fn, 5)
    assert caught_other is False
    assert err_other is None

def test_retrace_counter():
    def sample_fn(x):
        return x

    inputs = [
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        np.zeros((2, 8)),
        np.zeros((2, 8)),
    ]
    count = trace_and_count_retraces(sample_fn, inputs)
    assert count == 3
