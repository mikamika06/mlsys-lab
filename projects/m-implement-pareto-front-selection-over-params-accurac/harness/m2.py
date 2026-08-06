import ref


def check(workdir):
    from pareto.metrics import measure_retention

    out = {"retention_match": 0.0}
    want = ref.measure_retention(ref.STUDENT_LOGITS, ref.TEACHER_LOGITS, ref.TARGETS)
    try:
        got = measure_retention(ref.STUDENT_LOGITS, ref.TEACHER_LOGITS, ref.TARGETS)
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {str(e)[:100]}"
        return out

    if isinstance(got, (int, float)) and abs(got - want) < 1e-5:
        out["retention_match"] = 1.0
    else:
        out["_note"] = f"got retention {got}, expected {want}"
    return out
