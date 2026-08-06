def calculate_expected_acceptance(acceptance_probs: list[float], gamma: int) -> dict:
    """Calculate cumulative acceptance probabilities and expected accepted tokens."""
    probs = acceptance_probs[:gamma]
    cum_probs = []
    running = 1.0
    for p in probs:
        running *= p
        cum_probs.append(running)
    expected_accepted = 1.0 + sum(cum_probs)
    return {
        "gamma": gamma,
        "cum_probs": cum_probs,
        "expected_accepted": expected_accepted,
    }
