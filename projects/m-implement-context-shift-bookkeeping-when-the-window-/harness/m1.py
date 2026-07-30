import ref


def check(workdir):
    from ctxshift import simulate

    out = {"cases_matched": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, (n_ctx, n_keep, n_tokens) in enumerate(ref.CASES):
        want = ref.simulate(n_ctx, n_keep, n_tokens)
        got = simulate(n_ctx, n_keep, n_tokens)
        norm = ({k: got.get(k) for k in ("resident", "evicted", "shift_events")}
                if isinstance(got, dict) else got)
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = (f"case {i} (n_ctx={n_ctx}, n_keep={n_keep}, n_tokens={n_tokens}): "
                             f"got {norm}, reference {want}")
    out["cases_matched"] = float(ok)
    return out
