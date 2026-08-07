import ref
from shapes.evaluator import evaluate_rangedim, evaluate_enumerated


def check(workdir):
    out = {"numerics_matched": 0.0, "evaluations_correct": 0.0}
    try:
        r_out = evaluate_rangedim(("seq", 16, 64, 128), ref.SAMPLE_INPUTS)
        e_out = evaluate_enumerated([16, 32, 64, 128], ref.SAMPLE_INPUTS)

        if len(r_out) == len(e_out) and all(abs(a - b) < 1e-5 for a, b in zip(r_out, e_out)):
            out["numerics_matched"] = 1.0

        evals_ok = 0
        if len(r_out) == len(ref.SAMPLE_INPUTS):
            evals_ok += 1
        if len(e_out) == len(ref.SAMPLE_INPUTS):
            evals_ok += 1
        if out["numerics_matched"] == 1.0:
            evals_ok += 1

        out["evaluations_correct"] = float(evals_ok)
    except Exception as e:
        out["_note"] = f"Milestone 1 error: {type(e).__name__}: {str(e)[:120]}"
    return out
