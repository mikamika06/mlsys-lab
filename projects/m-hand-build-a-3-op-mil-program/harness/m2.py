import ref


def check(workdir):
    from miltool.passes import simulate_sdpa_pass

    out = {"pass_effect_matched": 0.0}
    try:
        seq_lens = [512, 1024, 2048]
        matched = True
        for sl in seq_lens:
            got = simulate_sdpa_pass(sl)
            want = ref.generate_pass_oracle(sl)
            if got != want:
                matched = False
                out["_note"] = f"seq_len {sl}: got {got}, expected {want}"
                break
        if matched:
            out["pass_effect_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)[:120]}"
    return out
