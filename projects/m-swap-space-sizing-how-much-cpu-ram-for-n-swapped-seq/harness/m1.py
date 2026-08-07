import ref


def check(workdir):
    from swapspace.sizing import (
        compute_sequence_swap_bytes,
        compute_total_swap_bytes,
    )

    out = {"sizing_matched": 0.0, "total_configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, (cfg, workload) in enumerate(zip(ref.CONFIGS, ref.WORKLOADS)):
        want_seqs = [ref.compute_sequence_swap_bytes(cfg, t) for t in workload]
        got_seqs = [compute_sequence_swap_bytes(cfg, t) for t in workload]
        want_tot = ref.compute_total_swap_bytes(cfg, workload)
        got_tot = compute_total_swap_bytes(cfg, workload)

        if want_seqs == got_seqs and want_tot == got_tot:
            ok += 1
        elif "_note" not in out:
            out["_note"] = (
                f"cfg {i}: got tot={got_tot}, want tot={want_tot}; got seqs={got_seqs[:2]}, want={want_seqs[:2]}"
            )

    out["sizing_matched"] = float(ok)
    return out
