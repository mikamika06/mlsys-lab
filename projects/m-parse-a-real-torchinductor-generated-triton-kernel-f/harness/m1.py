import ref


def check(workdir):
    from inductor_parse.parser import parse_kernel_config

    out = {"kernels_parsed": 0.0, "total_kernels": float(len(ref.SAMPLE_KERNELS))}
    ok = 0

    for i, code in enumerate(ref.SAMPLE_KERNELS):
        want = ref.parse_kernel_config(code)
        try:
            got = parse_kernel_config(code)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"kernel {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"kernel {i} raised {type(e).__name__}: {e}"

    out["kernels_parsed"] = float(ok)
    return out
