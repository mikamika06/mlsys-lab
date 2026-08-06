import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from schema_runner.benchmark import measure_throughput

    schema = ref.SCHEMAS[0]
    res = measure_throughput(
        ref.VOCAB, ref.EOS_ID, schema, ref.mock_logits_fn, max_tokens=20
    )

    valid = (
        isinstance(res, dict)
        and "unconstrained_tok_s" in res
        and "constrained_tok_s" in res
        and "speedup_ratio" in res
    )

    ratio_ok = valid and res["speedup_ratio"] > 0.0

    return {
        "benchmark_valid": 1.0 if valid else 0.0,
        "ratio_computed": 1.0 if ratio_ok else 0.0,
    }
