import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    import trtep.audit as audit
    import trtep.rewriter as rewriter

    m = {"coverage_ratio_ok": 0.0, "cpu_fallback_minimized": 0.0}
    try:
        g = ref.build_benchmark_graph()
        rewritten = rewriter.rewrite_graph(g, ref.DEFAULT_SUPPORTED_OPS)
        subs = audit.partition_graph(rewritten, ref.DEFAULT_SUPPORTED_OPS)
    except Exception:
        return m

    total_nodes = len(rewritten.nodes)
    if total_nodes == 0:
        return m

    trt_nodes = sum(len(s.nodes) for s in subs if s.provider == "TensorRT")
    cpu_nodes = sum(len(s.nodes) for s in subs if s.provider == "CPU")
    coverage = trt_nodes / total_nodes

    if coverage >= 0.85:
        m["coverage_ratio_ok"] = 1.0

    if cpu_nodes <= 2:
        m["cpu_fallback_minimized"] = 1.0

    return m
