import sys

import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from moeload.benchmark import run_benchmark_session
    from moeload.server import MoEServer

    out = {"traces_matched": 0.0}
    ok = 0

    for i, workload in enumerate(ref.WORKLOADS):
        server = MoEServer()
        ref_server = ref.RefMoEServer()

        got_traces = run_benchmark_session(server, workload, concurrency=i + 1)
        want_traces = ref.ref_run_benchmark_session(ref_server, workload, concurrency=i + 1)

        if len(got_traces) == len(want_traces):
            matched = True
            for g, w in zip(got_traces, want_traces):
                if (
                    g.get("prompt_tokens") != w.get("prompt_tokens")
                    or g.get("decode_tokens") != w.get("decode_tokens")
                    or abs(g.get("ttft_ms", 0.0) - w.get("ttft_ms", 0.0)) > 1e-4
                    or abs(g.get("total_time_ms", 0.0) - w.get("total_time_ms", 0.0)) > 1e-4
                ):
                    matched = False
                    break
            if matched:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"workload {i} trace mismatch: got {got_traces[0]} vs ref {want_traces[0]}"
        elif "_note" not in out:
            out["_note"] = f"workload {i} returned {len(got_traces)} items, expected {len(want_traces)}"

    out["traces_matched"] = float(ok)
    return out
