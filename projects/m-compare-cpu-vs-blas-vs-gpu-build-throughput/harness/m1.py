import ref

def check(workdir):
    from backperf.throughput import compare_throughput
    fixtures = ref.get_throughput_fixtures()
    want = compare_throughput(fixtures)
    try:
        got = compare_throughput(fixtures)
    except Exception as e:
        return {"throughput_matched": 0.0, "_note": f"raised {e}"}

    if got == want:
        return {"throughput_matched": 1.0}
    else:
        return {"throughput_matched": 0.0, "_note": f"got {got}, want {want}"}
