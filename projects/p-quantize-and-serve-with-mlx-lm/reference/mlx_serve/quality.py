import ref

def measure_quality(orig, quant) -> float:
    return ref.compute_perplexity_diff(orig, quant)
