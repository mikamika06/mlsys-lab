import sys

sys.path.insert(0, ".")
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from evalrec.metrics import rank_quant_candidates
    except Exception as e:
        return {
            "ranks_matched": 0.0,
            "disagreement_detected": 0.0,
            "_note": f"Import error: {type(e).__name__}: {e}",
        }

    teacher_data, candidates_data = ref.get_m1_test_cases()

    try:
        want = ref.rank_quant_candidates(teacher_data, candidates_data)
        got = rank_quant_candidates(teacher_data, candidates_data)
    except Exception as e:
        return {
            "ranks_matched": 0.0,
            "disagreement_detected": 0.0,
            "_note": f"Execution error: {type(e).__name__}: {e}",
        }

    ranks_matched = 1.0 if got == want else 0.0
    disagreement_detected = (
        1.0
        if (
            got.get("rank_disagreement") is True
            and want.get("rank_disagreement") is True
        )
        else 0.0
    )

    out = {
        "ranks_matched": ranks_matched,
        "disagreement_detected": disagreement_detected,
    }
    if not ranks_matched:
        out["_note"] = f"Got {got}, expected {want}"[:120]

    return out
