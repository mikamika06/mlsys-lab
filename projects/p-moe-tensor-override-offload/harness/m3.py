def check(workdir):
    import ref
    from moe_offload.offload import MoEOffloader
    m = {"latency_profile_ok": 0.0}
    sizes, _, base_latency, _, _ = ref.get_sample_data()
    offloader = MoEOffloader(sizes, 600)
    offloaded = {3, 4, 5}
    lat = offloader.evaluate_latency(offloaded, base_latency, penalty_factor=2.0)
    expected = 147.0
    if abs(lat - expected) < 1e-5:
        m["latency_profile_ok"] = 1.0
    return m
