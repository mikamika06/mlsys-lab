"""Analysis utilities for comparing eager and compiled execution."""

def compare_execution(eager_trace, compiled_trace):
    eager_ops = eager_trace.get("ops", 0)
    compiled_ops = compiled_trace.get("ops", 0)
    eager_size = eager_trace.get("size", 0)
    compiled_size = compiled_trace.get("size", 0)
    size_ratio = compiled_size / (eager_size if eager_size > 0 else 1.0)
    op_ratio = compiled_ops / (eager_ops if eager_ops > 0 else 1.0)
    return {
        "eager_ops": eager_ops,
        "compiled_ops": compiled_ops,
        "size_ratio": size_ratio,
        "op_ratio": op_ratio
    }
