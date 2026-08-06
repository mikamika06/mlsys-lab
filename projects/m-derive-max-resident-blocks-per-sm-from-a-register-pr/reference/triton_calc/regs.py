import math


def effective_regs(regs_per_thread: int, granularity: int) -> int:
    """Compute effective registers per thread after hardware alignment."""
    return math.ceil(regs_per_thread / granularity) * granularity
