def differences(a, b):
    """Configuration keys where two runs disagree, sorted.

    The file a row came from is not a difference, and neither is the model
    path — the same model reached by two paths is the same model.
    """
    raise NotImplementedError


def controlled(rows, axis):
    """(i, j) for every pair that differs in `axis` and nothing else."""
    raise NotImplementedError


def confounded(rows, axis):
    """(i, j, other_keys) for pairs that differ in `axis` and in something
    else too — the comparisons somebody is about to quote by mistake."""
    raise NotImplementedError
