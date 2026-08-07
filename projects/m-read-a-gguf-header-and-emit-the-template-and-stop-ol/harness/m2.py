import ref


def check(workdir):
    from ggufparse.quant import profile_quants
    out = {"quant_matched": 0.0}
    try:
        res = profile_quants(ref.BASE_SIZE, ref.BASE_TOK_S)
        if isinstance(res, dict) and "Q8_0" in res and "Q4_K_M" in res:
            out["quant_matched"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
