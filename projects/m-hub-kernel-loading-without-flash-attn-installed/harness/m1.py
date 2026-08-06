import ref


def check(workdir):
    from hf_attn.router import resolve_backend, dispatch_attention

    out = {"backends_matched": 0.0}
    ok = 0
    total = len(ref.CONFIGS)

    for i, item in enumerate(ref.CONFIGS):
        cfg = item["config"]
        backends = item["backends"]
        want_res = ref.resolve_backend(cfg, backends)
        want_disp = ref.dispatch_attention(cfg, None, None, None, backends)

        try:
            got_res = resolve_backend(cfg, backends)
            got_disp = dispatch_attention(cfg, None, None, None, backends)
            if got_res == want_res and got_disp == want_disp:
                ok += 1
            else:
                out["_note"] = f"case {i}: expected ({want_res}, {want_disp}), got ({got_res}, {got_disp})"
                break
        except Exception as e:
            out["_note"] = f"case {i} raised exception: {e}"
            break

    if ok == total:
        out["backends_matched"] = 1.0
    return out
