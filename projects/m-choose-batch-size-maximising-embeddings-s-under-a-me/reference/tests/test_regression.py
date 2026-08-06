from embedopt.truncation import truncate_sequence


def test_truncation_overflow_error():
    tokens = list(range(32))
    num_ctx = 16
    try:
        truncate_sequence(tokens, num_ctx, policy="error")
        assert False, "Expected ValueError when sequence exceeds num_ctx under 'error' policy"
    except ValueError:
        pass
