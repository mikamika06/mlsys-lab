import ref
import numpy as np

def check(workdir):
    from embedrunner.detect import is_l2_normalized
    from embedrunner.safety import analyze_model_mixing

    out = {"normalized_detected": 0.0, "mixing_demonstrated": 0.0}

    norm_data = ref.generate_embeddings(normalized=True, count=5)
    unnorm_data = ref.generate_embeddings(normalized=False, count=5)

    try:
        r1 = is_l2_normalized(norm_data)
        r2 = is_l2_normalized(unnorm_data)
        if r1 is True and r2 is False:
            out["normalized_detected"] = 1.0
        else:
            out["_note"] = f"is_l2_normalized gave incorrect results: normal={r1}, unnormal={r2}"
    except Exception as e:
        out["_note"] = f"is_l2_normalized exception: {type(e).__name__}: {str(e)[:120]}"

    try:
        m_result = analyze_model_mixing()
        if m_result is not None:
            out["mixing_demonstrated"] = 1.0
        else:
            out["_note"] = "analyze_model_mixing returned None"
    except Exception as e:
        out["_note"] = f"analyze_model_mixing exception: {type(e).__name__}: {str(e)[:120]}"

    return out
