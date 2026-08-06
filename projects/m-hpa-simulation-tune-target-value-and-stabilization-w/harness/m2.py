import ref


def check(workdir):
    from hpa.sim import diagnose_thrash
    from hpa.affinity import evaluate_session_affinity

    out = {"diagnose_match": 0.0, "affinity_match": 0.0}

    diag_got = diagnose_thrash(ref.TIMELINES[0])
    diag_want = ref.diagnose_thrash(ref.TIMELINES[0])
    if diag_got == diag_want:
        out["diagnose_match"] = 1.0

    random.seed(42)
    aff_got = evaluate_session_affinity(ref.SESSIONS[0], 4, "random")
    random.seed(42)
    aff_want = ref.evaluate_session_affinity(ref.SESSIONS[0], 4, "random")

    if abs(aff_got - aff_want) < 1e-5:
        out["affinity_match"] = 1.0

    return out
