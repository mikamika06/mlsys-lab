import ref

def check(workdir):
    from recomp.starvation import derive_max_num_batched_tokens
    out = {"token_limit_match": 0.0}
    try:
        got = derive_max_num_batched_tokens(ref.WAITING, ref.RUNNING, ref.TARGET_RATIO)
        want = ref.derive_max_num_batched_tokens(ref.WAITING, ref.RUNNING, ref.TARGET_RATIO)
        if got == want:
            out["token_limit_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"raised error: {type(e).__name__}: {str(e)[:120]}"
    return out
