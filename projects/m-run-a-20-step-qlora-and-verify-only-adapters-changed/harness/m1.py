import ref

def check(workdir):
    from qlora.trainer import run_qlora_steps
    out = {"steps_run": 0.0}
    init_m = ref.get_model()
    data = ref.get_data()
    try:
        final_m = run_qlora_steps(init_m, data, steps=20)
        if final_m is not None:
            out["steps_run"] = 20.0
    except Exception as e:
        out["_note"] = f"run_qlora_steps raised {type(e).__name__}: {str(e)[:100]}"
    return out
