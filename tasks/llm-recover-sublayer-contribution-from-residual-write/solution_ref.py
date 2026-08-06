def recover_sublayer_contribution(in_, out_):
    """Return the sublayer contribution from residual write."""
    if isinstance(in_[0], list):
        return [[o - i for i, o in zip(row_in, row_out)] for row_in, row_out in zip(in_, out_)]
    else:
        return [o - i for i, o in zip(in_, out_)]
