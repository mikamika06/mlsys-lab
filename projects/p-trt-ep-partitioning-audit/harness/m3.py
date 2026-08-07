import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    import trtep.audit as audit
    import trtep.rewriter as rewriter

    m = {"rewrite_api_ok": 0.0, "partitions_reduced": 0.0}
    try:
        g = ref.build_benchmark_graph()
        rewritten = rewriter.rewrite_graph(g, ref.DEFAULT_SUPPORTED_OPS)
    except Exception:
        return m

    if not hasattr(rewritten, "nodes") or len(rewritten.nodes) != len(g.nodes):
        return m

    m["rewrite_api_ok"] = 1.0

    orig_subs = audit.partition_graph(g, ref.DEFAULT_SUPPORTED_OPS)
    new_subs = audit.partition_graph(rewritten, ref.DEFAULT_SUPPORTED_OPS)

    if len(new_subs) < len(orig_subs) and len(new_subs) <= 2:
        m["partitions_reduced"] = 1.0

    return m
