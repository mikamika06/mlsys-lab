import ref


def check(workdir):
    from kvcache.server import measure_hit_rate

    _, traces = ref.get_fixtures()
    ref_rate = ref.measure_hit_rate(traces, 100, 16)
    got_rate = measure_hit_rate(traces, 100, 16)

    return {
        "hit_rate_matches": 1.0 if abs(ref_rate - got_rate) < 1e-5 else 0.0
    }
