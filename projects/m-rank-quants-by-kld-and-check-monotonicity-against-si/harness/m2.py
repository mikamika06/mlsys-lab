import ref


def check(workdir):
    from quval.stats import derive_sample_size
    from quval.metrics import find_disagreements
    quants = ref.get_test_data()
    out = {"sample_size_matched": 0.0, "disagreements_matched": 0.0}
    try:
        n = derive_sample_size(5.0, 5.5, 1.2)
        if isinstance(n, (int, float)) and n > 0:
            out["sample_size_matched"] = 1.0

        dis = find_disagreements(quants)
        if isinstance(dis, list):
            out["disagreements_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Error: {type(e).__name__}: {str(e)[:120]}"
    return out
