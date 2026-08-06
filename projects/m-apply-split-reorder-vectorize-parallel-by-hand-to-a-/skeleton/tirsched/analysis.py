def print_tir_loop_nest(tir_mod):
    """Format TIR module into a readable loop nest string representation."""
    raise NotImplementedError


def map_axes_to_transforms(loop_nest_str):
    """Parse printed loop nest string and map each axis name to its transformation."""
    raise NotImplementedError
