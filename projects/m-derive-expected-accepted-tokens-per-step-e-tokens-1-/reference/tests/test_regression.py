import speculative.theory
import speculative.sampling
import speculative.trace

def test_theory_invariant():
    val = speculative.theory.expected_accepted_tokens(0.5, 3)
    assert isinstance(val, float)
    assert val >= 0.0

def test_sampling_output():
    res = speculative.sampling.modified_rejection_sampling([0.2, 0.3, 0.5], [0.1, 0.4, 0.5], 42)
    assert len(res) == 3

def test_trace_output():
    res = speculative.trace.generate_trace([0.2, 0.3, 0.5], [0.1, 0.4, 0.5], 42, 2)
    assert len(res) == 2
