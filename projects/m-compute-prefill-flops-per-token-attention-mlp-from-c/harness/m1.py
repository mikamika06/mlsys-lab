import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from roofline.flops import compute_prefill_flops_per_token

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0

    for i, cfg in enumerate(ref.CONFIGS):
        for seq_len in [512, 2048]:
            want = ref.ref_compute_prefill_flops_per_token(cfg, seq_len)
            try:
                got = compute_prefill_flops_per_token(cfg, seq_len)
            except Exception as e:
                out["_note"] = f"config {i} raised exception: {type(e).__name__}: {str(e)}"
                return out

            rel_err = abs(got - want) / want if want != 0 else 0.0
            if rel_err < 1e-5:
                ok += 0.5
            elif "_note" not in out:
                out["_note"] = f"config {i} seq {seq_len}: got {got}, want {want} (rel_err: {rel_err:.6f})"

    out["configs_matched"] = float(ok)
    return out
