def per_example_loop(fn, x_batch, axis=0):
    """Applies fn to each single example slice along axis and stacks results."""
    raise NotImplementedError


def verify_vmap_matches(fn_single, fn_batched, x_batch, axis=0):
    """Compares per_example_loop(fn_single) against fn_batched and returns max absolute error."""
    raise NotImplementedError
