def drop_token_0(attention_probs):
    """Drop token 0 from all queries > 0 and renormalize."""
    raise NotImplementedError

def measure_blowup(original_probs, ablated_probs):
    """Measure the maximum absolute difference between the probabilities."""
    raise NotImplementedError
