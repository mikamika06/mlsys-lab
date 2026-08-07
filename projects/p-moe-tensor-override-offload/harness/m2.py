def check(workdir):
    import ref
    from moe_offload.offload import MoEOffloader
    m = {"rules_ok": 0.0}
    sizes, traces, _, _, _ = ref.get_sample_data()
    offloader = MoEOffloader(sizes, 600)
    freqs = offloader.measure_frequencies(traces)
    offloaded = offloader.compute_rules(freqs, 600)
    expected_offloaded = {3, 4, 5}
    if offloaded == expected_offloaded:
        m["rules_ok"] = 1.0
    return m
