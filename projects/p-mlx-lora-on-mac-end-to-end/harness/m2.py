def check(workdir):
    import ref
    from lora_pipe import engine
    m = {"loss_converged": 0.0}
    data = ref.get_sample_data()
    try:
        out = engine.run_lora(data, steps=4, lr=0.01)
        if isinstance(out, dict) and "losses" in out and len(out["losses"]) == 4:
            if out["losses"][-1] <= out["losses"][0] + 0.5:
                m["loss_converged"] = 1.0
    except Exception:
        pass
    return m
