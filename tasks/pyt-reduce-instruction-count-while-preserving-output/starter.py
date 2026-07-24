def clamp(x, lo, hi):
    """Bound x into [lo, hi] (lo <= hi): lo if x < lo, hi if x > hi, else x.

    Keep the bytecode short: len(list(dis.get_instructions(clamp))) must be
    small, so prefer a direct expression over an explicit branch tree.
    """
    raise NotImplementedError('your code here')
