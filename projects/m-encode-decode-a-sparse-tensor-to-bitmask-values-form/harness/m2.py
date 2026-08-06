import ref


def check(workdir):
    from sparsecoder.analysis import breakeven_sparsity, measure_byte_savings

    shape = (32, 32)
    ref_be = ref.breakeven_sparsity(shape, dtype_bytes=2)
    got_be = breakeven_sparsity(shape, dtype_bytes=2)

    be_match = 1 if ref_be == got_be else 0

    sparsities = [0.1, 0.5, 0.9]
    ref_savings = ref.measure_byte_savings(shape, sparsities, dtype_bytes=2, block_size=8)
    got_savings = measure_byte_savings(shape, sparsities, dtype_bytes=2, block_size=8)

    savings_match = 1
    for s in sparsities:
        if s not in got_savings or got_savings[s].get("sparse_bytes") != ref_savings[s].get("sparse_bytes"):
            savings_match = 0
            break

    return {
        "breakeven_match": int(be_match),
        "savings_match": int(savings_match)
    }
