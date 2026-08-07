def check(workdir):
    import ref
    from moe_offload.offload import MoEOffloader
    m = {"freq_ok": 0.0}
    sizes, traces, _, _, _ = ref.get_sample_data()
    offloader = MoEOffloader(sizes, 500)
    freqs = offloader.measure_frequencies(traces)
    expected = ref.np.array([1.0, 1.0, 0.6, 0.2, 0.0, 0.0])
    if ref.np.allclose(freqs, expected):
        m["freq_ok"] = 1.0
    return m
