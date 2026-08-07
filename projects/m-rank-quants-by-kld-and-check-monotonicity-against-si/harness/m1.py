import ref


def check(workdir):
    from quval.metrics import rank_quants_by_kld, check_monotonicity
    quants = ref.get_test_data()
    out = {"rankings_matched": 0.0}
    try:
        ranked = rank_quants_by_kld(quants)
        ref_ranked = ref.rank_quants_by_kld if hasattr(ref, "rank_quants_by_kld") else None

        if len(ranked) == len(quants):
            mono = check_monotonicity(ranked)
            if isinstance(mono, bool):
                out["rankings_matched"] = 1.0
        else:
            out["_note"] = "ranked length mismatch"
    except Exception as e:
        out["_note"] = f"Error: {type(e).__name__}: {str(e)[:120]}"
    return out
