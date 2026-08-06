import ref


def check(workdir):
    from dynshape.count import count_graphs

    def sample_func(x):
        return x * 2.0 + 1.0

    shapes = [(2, 16), (4, 16), (8, 16), (2, 32)]
    ref_count = ref.count_graphs(sample_func, shapes)
    try:
        learner_count = count_graphs(sample_func, shapes)
    except Exception:
        learner_count = -999

    out = {"graph_count_match": 1.0 if learner_count == ref_count else 0.0}
    if learner_count != ref_count:
        out["_note"] = f"got graph count {learner_count}, expected {ref_count}"
    return out
