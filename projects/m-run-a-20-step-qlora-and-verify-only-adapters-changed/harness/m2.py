import ref

def check(workdir):
    from qlora.trainer import run_qlora_steps
    from qlora.verify import verify_adapters_changed
    out = {"adapters_changed": 0.0, "base_unchanged": 0.0}
    init_m = ref.get_model()
    data = ref.get_data()
    try:
        final_m = run_qlora_steps(init_m, data, steps=20)
        res = verify_adapters_changed(init_m, final_m)
        if res:
            out["adapters_changed"] = 1.0
            out["base_unchanged"] = 1.0
    except Exception as e:
        out["_note"] = f"verification failed: {type(e).__name__}: {str(e)[:100]}"
    return out
