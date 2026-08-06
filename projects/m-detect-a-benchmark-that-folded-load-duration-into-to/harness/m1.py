import ref

def check(workdir):
    from runner.audit import parse_benchmark
    bms = ref.get_test_benchmarks()
    ok = 0
    for b in bms:
        try:
            res = parse_benchmark(b)
            if res["total_tokens"] == b["total_tokens"] and res["generation_duration"] == b["generation_duration"]:
                ok += 1
        except Exception:
            pass
    return {"parsed_correctly": 1.0 if ok == len(bms) else 0.0}
