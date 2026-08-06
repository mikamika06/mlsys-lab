import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from brgemm.roofline import analyze_roofline, compute_memory_traffic

    max_err = 0.0

    for cfg in ref.CONFIGS:
        want_traffic = ref.compute_memory_traffic(
            cfg["M"], cfg["N"], cfg["K"], cfg["mb"], cfg["nb"], cfg["kb"]
        )
        got_traffic = compute_memory_traffic(
            cfg["M"], cfg["N"], cfg["K"], cfg["mb"], cfg["nb"], cfg["kb"]
        )
        err = abs(got_traffic - want_traffic) / max(1.0, float(want_traffic))
        if err > max_err:
            max_err = err

        want_roof = ref.analyze_roofline(
            cfg["M"], cfg["N"], cfg["K"], cfg["mb"], cfg["nb"], cfg["kb"],
            cfg["peak"], cfg["bw"], cfg["runtime"]
        )
        got_roof = analyze_roofline(
            cfg["M"], cfg["N"], cfg["K"], cfg["mb"], cfg["nb"], cfg["kb"],
            cfg["peak"], cfg["bw"], cfg["runtime"]
        )

        for k, want_val in want_roof.items():
            got_val = got_roof.get(k, 0.0)
            err_k = abs(got_val - want_val) / max(1e-9, float(abs(want_val)))
            if err_k > max_err:
                max_err = err_k

    out = {"rel_err": float(max_err)}
    if max_err > 0.01:
        out["_note"] = f"max relative error was {max_err:.4f}"
    return out
