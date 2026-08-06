import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"solver_matched": 0.0, "blocks_matched": 0.0}
    try:
        from vllm_budget.blocks import predict_num_gpu_blocks
        from vllm_budget.solver import max_context_length
    except Exception as e:
        out["_note"] = f"Failed to import solver/blocks functions: {e}"
        return out

    s_ok = 0
    for fix in ref.SOLVER_FIXTURES:
        want = ref.ref_max_context_length(**fix)
        try:
            got = max_context_length(**fix)
            if got == want:
                s_ok += 1
            elif "_note" not in out:
                out["_note"] = f"max_context_length mismatch: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Error in max_context_length: {e}"

    b_ok = 0
    for fix in ref.BLOCKS_FIXTURES:
        want = ref.ref_predict_num_gpu_blocks(**fix)
        try:
            got = predict_num_gpu_blocks(**fix)
            if got == want:
                b_ok += 1
            elif "_note" not in out:
                out["_note"] = f"predict_num_gpu_blocks mismatch: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Error in predict_num_gpu_blocks: {e}"

    if s_ok == len(ref.SOLVER_FIXTURES):
        out["solver_matched"] = 1.0
    if b_ok == len(ref.BLOCKS_FIXTURES):
        out["blocks_matched"] = 1.0

    return out
