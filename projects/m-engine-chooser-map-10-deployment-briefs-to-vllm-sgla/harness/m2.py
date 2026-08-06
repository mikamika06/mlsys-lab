import ref


def check(workdir):
    from chooser.matrix import get_feature_matrix
    out = {"matrix_match": 0.0}
    try:
        got = get_feature_matrix()
        want = ref.get_feature_matrix()
        if got == want:
            out["matrix_match"] = 1.0
        else:
            out["_note"] = "feature matrix does not match reference official docs mapping"
    except Exception as e:
        out["_note"] = f"error executing get_feature_matrix: {str(e)[:120]}"
    return out
