import ref


def check(workdir):
    from miltool.builder import build_three_op_program

    out = {"program_matched": 0.0}
    try:
        got = build_three_op_program()
        want = ref.generate_program_oracle()
        if got == want:
            out["program_matched"] = 1.0
        else:
            out["_note"] = f"got program {got}, expected {want}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)[:120]}"
    return out
