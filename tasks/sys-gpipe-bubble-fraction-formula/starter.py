def gpipe_bubble_fraction(microbatches: int, stages: int) -> float:
    """Fraction of GPipe's per-device time slots spent idle (the "bubble").

    `microbatches` -- number of microbatches per pipeline flush (m).
    `stages` -- number of pipeline stages / devices (p).
    Returns the idle fraction as a float in [0, 1). See task.md for the
    exact formula.
    """
    raise NotImplementedError('your code here')
