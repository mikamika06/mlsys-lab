def expected_tokens_per_step(accept_probs: list[float]) -> float:
    """Expected number of tokens emitted in one speculative decoding step.

    accept_probs[i] is the probability that draft position i is accepted,
    conditional on every earlier position having been accepted. The target
    model always emits exactly one extra token beyond the accepted prefix
    (a correction on rejection, or a bonus token if all are accepted).
    Returns the expectation as a Python float.
    """
    raise NotImplementedError('your code here')
