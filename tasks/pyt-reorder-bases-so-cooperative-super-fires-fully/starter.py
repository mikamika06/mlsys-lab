def reorder_bases(bases):
    # TODO: keeps the broken input ordering. In cases where a subclass appears
    # after its parent, Python rejects the MRO or the cooperative chain cannot
    # execute in the intended order.
    return tuple(bases)
