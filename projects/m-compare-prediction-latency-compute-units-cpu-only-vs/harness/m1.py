import ref


def check(workdir):
    from coreflow.latency import compare_latency
    pkg = ref.generate_mock_package()
    try:
        res = compare_latency(pkg, {"input": [1, 3, 224, 224]})
        out = {
            "latency_compared": 1.0 if isinstance(res, dict) and "speedup_ratio" in res and res["valid"] else 0.0
        }
        return out
    except Exception as e:
        return {"latency_compared": 0.0, "_note": f"failed: {str(e)[:120]}"}
