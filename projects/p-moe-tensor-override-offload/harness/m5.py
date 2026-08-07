def check(workdir):
    import ref
    from moe_offload.offload import MoEOffloader
    m = {"memory_speed_ok": 0.0}
    sizes, _, _, _, _ = ref.get_sample_data()
    offloader = MoEOffloader(sizes, 600)
    offloaded = {3, 4, 5}
    lat = 147.0
    ok = offloader.check_constraints(offloaded, 600, lat, 150.0)
    if ok:
        m["memory_speed_ok"] = 1.0
    return m
