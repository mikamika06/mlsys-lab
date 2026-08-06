def parse_stablehlo_op_counts(stablehlo_code):
    """Parse StableHLO code string and count occurrences of each operation."""
    raise NotImplementedError

def get_stablehlo_ops(jit_fn, args, flags=None):
    """Lower function under given flags and return StableHLO operation counts."""
    raise NotImplementedError

def diff_stablehlo_ops(jit_fn, args, flags_a=None, flags_b=None):
    """Diff StableHLO operation counts between two sets of compiler flags."""
    raise NotImplementedError
