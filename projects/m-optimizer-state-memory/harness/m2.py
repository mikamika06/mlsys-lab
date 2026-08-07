import sys
import torch
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"compiled_overhead_matched": 0.0, "zerograd_footprint_matched": 0.0}

    try:
        from optmem.compile import compile_optimizer_step, measure_step_memory_delta
        from optmem.zerograd import profile_zerograd_allocation
    except Exception as e:
        out["_note"] = f"Failed to import optmem modules: {e}"
        return out

    model = ref.get_test_model()
    inputs = ref.get_test_inputs()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        compiled_step = compile_optimizer_step(model, optimizer)
        delta = measure_step_memory_delta(model, optimizer, compiled_step, inputs)
        if (
            isinstance(delta, dict)
            and "eager_grad_alloc_bytes" in delta
            and "compiled_grad_alloc_bytes" in delta
            and "step_overhead_ratio" in delta
        ):
            out["compiled_overhead_matched"] = 1.0
        else:
            out["_note"] = f"Invalid step memory delta structure: {delta}"
    except Exception as e:
        out["_note"] = f"Error in compile memory delta: {e}"
        return out

    try:
        model_zg = ref.get_test_model()
        inputs_zg = ref.get_test_inputs()
        want_zg = ref.ref_profile_zerograd(model_zg, inputs_zg)

        model_test = ref.get_test_model()
        inputs_test = ref.get_test_inputs()
        got_zg = profile_zerograd_allocation(model_test, inputs_test)

        if (
            got_zg.get("zero_fill_bytes") == want_zg["zero_fill_bytes"]
            and got_zg.get("none_fill_bytes") == want_zg["none_fill_bytes"]
            and got_zg.get("allocated_bytes_saved") == want_zg["allocated_bytes_saved"]
            and got_zg.get("none_fill_count") == 0
        ):
            out["zerograd_footprint_matched"] = 1.0
        else:
            out["_note"] = f"Zerograd profile mismatch: want {want_zg}, got {got_zg}"
    except Exception as e:
        out["_note"] = f"Error in zerograd profiling: {e}"

    return out
