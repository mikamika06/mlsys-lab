def median(xs):
    raise NotImplementedError


def quartiles(xs):
    """(q1, median, q3), excluding the median from both halves when the count
    is odd."""
    raise NotImplementedError


def iqr(xs):
    raise NotImplementedError


def separable(a, b):
    """1 when the two inter-quartile ranges do not overlap, else 0.

    Distance between medians is not evidence. Three repetitions on a warm
    laptop can put two medians far apart with the spreads sitting on top of
    each other, and that is the case this function exists to refuse.
    """
    raise NotImplementedError


def spread(xs):
    """IQR as a fraction of the median."""
    raise NotImplementedError
