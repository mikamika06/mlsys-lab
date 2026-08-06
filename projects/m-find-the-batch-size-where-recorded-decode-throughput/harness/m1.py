import ref


def check(workdir):
    from decode_prof.analysis import find_crossover_batch_size
    out = {"crossover_matched": 0.0}
    try:
        got = find_crossover_batch_size(ref.BATCH_SIZES, ref.TPUTS, ref.BW_REAL, ref.PEAK_BW)
        if int(got) == int(ref.CROSSOVER):
            out["crossover_matched"] = 1.0
        else:
            out["_note"] = f"got crossover batch size {got}, expected {ref.CROSSOVER}"
    except Exception as e:
        out["_note"] = f"error in find_crossover_batch_size: {type(e).__name__}: {str(e)[:120]}"
    return out
