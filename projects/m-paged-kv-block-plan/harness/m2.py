import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from kvplan.bench import generate_throughput_report
        from kvplan.sweep import run_request_rate_sweep
    except Exception as e:
        return {"throughput_ratio": 0.0, "sweeps_matched": 0, "_note": f"Import failed: {e}"}

    rep = generate_throughput_report(50, 2500, 2500, 10.0)
    ratio = rep.get("throughput_ratio", 0.0)

    rates = [1, 2, 5]
    want_sweep = ref.run_request_rate_sweep(rates, 16, 50)
    got_sweep = run_request_rate_sweep(rates, 16, 50)

    sweeps_matched = 1 if got_sweep == want_sweep else 0

    return {
        "throughput_ratio": float(ratio),
        "sweeps_matched": float(sweeps_matched)
    }
