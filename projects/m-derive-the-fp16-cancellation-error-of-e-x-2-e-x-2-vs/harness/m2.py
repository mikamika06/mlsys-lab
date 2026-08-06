import ref

def check(workdir):
    from variance.benchmark import analyze_bandwidth_advantage
    _, benchmarks = ref.generate_fixtures()
    got = analyze_bandwidth_advantage(benchmarks)
    want = ref.analyze_bandwidth_reference(benchmarks)
    ratio = float(got / (want + 1e-9))
    return {"bandwidth_ratio": ratio}
