import sys
sys.path.insert(0, ".")
from medusa.sampling import simulate_speculative_sampling

def test_typical_exceeds_strict_accepted_length():
    typ = simulate_speculative_sampling(None, None, "typical")
    st = simulate_speculative_sampling(None, None, "strict")
    assert typ > st, f"typical accepted length ({typ}) must exceed strict ({st})"

def test_sampling_lengths_positive():
    typ = simulate_speculative_sampling(None, None, "typical")
    st = simulate_speculative_sampling(None, None, "strict")
    assert typ > 0 and st > 0, "accepted lengths must be strictly positive"
