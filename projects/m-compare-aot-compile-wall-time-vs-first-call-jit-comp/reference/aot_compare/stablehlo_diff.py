import re

def parse_stablehlo_op_counts(stablehlo_code):
    """Parse StableHLO code string and count occurrences of each operation."""
    matches = re.findall(r"stablehlo\.([a-zA-Z0-9_]+)", stablehlo_code)
    counts = {}
    for op in matches:
        counts[op] = counts.get(op, 0) + 1
    return counts

def get_stablehlo_ops(jit_fn, args, flags=None):
    """Lower function under given flags and return StableHLO operation counts."""
    flags = flags or {}
    try:
        lowered = jit_fn.lower(*args, flags=flags)
    except TypeError:
        lowered = jit_fn.lower(*args)
    text = lowered.as_text(dialect="stablehlo")
    return parse_stablehlo_op_counts(text)

def diff_stablehlo_ops(jit_fn, args, flags_a=None, flags_b=None):
    """Diff StableHLO operation counts between two sets of compiler flags."""
    ops_a = get_stablehlo_ops(jit_fn, args, flags_a)
    ops_b = get_stablehlo_ops(jit_fn, args, flags_b)
    all_ops = sorted(set(ops_a.keys()) | set(ops_b.keys()))
    return {op: ops_b.get(op, 0) - ops_a.get(op, 0) for op in all_ops}
