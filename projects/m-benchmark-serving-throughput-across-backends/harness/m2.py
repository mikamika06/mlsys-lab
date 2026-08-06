import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.ref as ref


def check(workdir):
    from serving.benchmark import generate_tradeoff_report, run_benchmark_pass

    out = {"throughput_ratio": 0.0, "report_structure_valid": 0.0}

    results = []
    workloads = ref.get_test_workloads()
    backends = ["FLASH_ATTN", "XFORMERS"]

    for wl in workloads:
        for b in backends:
            res = run_benchmark_pass(b, wl["batch_size"], wl["prompt_len"], wl["gen_len"])
            results.append(res)

    report = generate_tradeoff_report(results)

    if isinstance(report, dict) and "backends" in report and "FLASH_ATTN" in report["backends"] and "XFORMERS" in report["backends"]:
        out["report_structure_valid"] = 1.0
    else:
        out["_note"] = "Invalid tradeoff report structure"
        return out

    fa2_throughputs = [r["throughput_tok_s"] for r in report["backends"]["FLASH_ATTN"]]
    xformers_throughputs = [r["throughput_tok_s"] for r in report["backends"]["XFORMERS"]]

    avg_fa2 = sum(fa2_throughputs) / len(fa2_throughputs)
    avg_xformers = sum(xformers_throughputs) / len(xformers_throughputs)

    ratio = avg_fa2 / avg_xformers if avg_xformers > 0 else 0.0
    out["throughput_ratio"] = round(float(ratio), 3)

    return out
