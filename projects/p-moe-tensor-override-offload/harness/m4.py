def check(workdir):
    import ref
    from moe_offload.offload import MoEOffloader
    m = {"accuracy_ok": 0.0}
    sizes, _, _, ref_out, cand_out = ref.get_sample_data()
    offloader = MoEOffloader(sizes, 600)
    if offloader.verify_output(ref_out, cand_out, tol=1e-4):
        m["accuracy_ok"] = 1.0
    return m
