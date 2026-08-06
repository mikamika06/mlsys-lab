import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"rel_err": 1.0}
    try:
        from macroofline.roofline import (
            arithmetic_intensity,
            attainable_gflops,
            fit_empirical_roofline,
        )

        errs = []
        for run in ref.SAMPLE_RUNS:
            m, n, k = run["m"], run["n"], run["k"]
            itemsize = run.get("itemsize", 2)

            ref_ai = ref.arithmetic_intensity(m, n, k, itemsize)
            got_ai = arithmetic_intensity(m, n, k, itemsize)
            errs.append(abs(got_ai - ref_ai) / (abs(ref_ai) + 1e-12))

            ref_att = ref.attainable_gflops(ref_ai, 10000.0, ref.M4_MAX_PEAK_GBPS)
            got_att = attainable_gflops(got_ai, 10000.0, ref.M4_MAX_PEAK_GBPS)
            errs.append(abs(got_att - ref_att) / (abs(ref_att) + 1e-12))

        ref_fit = ref.fit_empirical_roofline(ref.SAMPLE_RUNS, ref.M4_MAX_PEAK_GBPS)
        got_fit = fit_empirical_roofline(ref.SAMPLE_RUNS, ref.M4_MAX_PEAK_GBPS)

        for key in ["peak_bandwidth_gbps", "empirical_peak_gflops", "knee_ai", "max_achieved_bw_gbps", "max_bw_utilization_pct"]:
            ref_val = ref_fit[key]
            got_val = got_fit[key]
            errs.append(abs(got_val - ref_val) / (abs(ref_val) + 1e-12))

        for rp, gp in zip(ref_fit["profiles"], got_fit["profiles"]):
            for key in ["ai", "achieved_gflops", "achieved_gbps"]:
                errs.append(abs(gp[key] - rp[key]) / (abs(rp[key]) + 1e-12))
            if gp["is_memory_bound"] != rp["is_memory_bound"]:
                errs.append(1.0)

        out["rel_err"] = max(errs) if errs else 1.0
    except Exception as e:
        out["_note"] = f"{type(e).__name__}: {e}"
    return out
