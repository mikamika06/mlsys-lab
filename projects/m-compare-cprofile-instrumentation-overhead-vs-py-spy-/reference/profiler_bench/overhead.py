import ref

def measure_overhead_ratio():
    """Measure ratio of cProfile instrumentation overhead to sampling overhead."""
    return ref.compute_reference_overhead()
