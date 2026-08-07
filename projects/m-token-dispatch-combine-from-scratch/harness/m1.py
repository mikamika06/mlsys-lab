import numpy as np
import ref


def check(workdir):
    from moe.dispatch import combine_tokens, dispatch_tokens

    out = {"dispatch_correctness": 0.0, "combine_correctness": 0.0}
    num_cases = len(ref.CASES_M1)
    dispatch_ok = 0
    combine_ok = 0

    for idx, case in enumerate(ref.CASES_M1):
        tokens, indices, weights, want_buffers, want_meta, want_combined = ref.run_reference_dispatch(case)
        try:
            got_buffers, got_meta = dispatch_tokens(tokens, indices, weights, case["num_experts"], case["capacity"])
            if np.allclose(got_buffers, want_buffers, atol=1e-5):
                dispatch_ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {idx}: dispatch buffers mismatch"

            got_combined = combine_tokens(got_buffers, got_meta)
            if np.allclose(got_combined, want_combined, atol=1e-5):
                combine_ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {idx}: combined output mismatch"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {idx} raised {type(e).__name__}: {str(e)[:100]}"

    out["dispatch_correctness"] = float(dispatch_ok / num_cases)
    out["combine_correctness"] = float(combine_ok / num_cases)
    return out
