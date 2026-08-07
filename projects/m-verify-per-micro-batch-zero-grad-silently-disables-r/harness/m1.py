import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"accum_steps_correct": 0.0, "grads_matched": 0.0}

    try:
        from gradaccum.accumulator import SimpleModel, run_correct_accumulation
    except Exception as e:
        out["_note"] = f"Failed to import accumulator module: {e}"
        return out

    weights = ref.make_weights(seed=42)
    mb_grads = ref.make_micro_batch_grads(num_mbs=12, seed=100)
    accum_steps = 4
    lr = 0.01

    try:
        model = SimpleModel(weights)
        got_grads = run_correct_accumulation(model, mb_grads, accum_steps, lr)
    except Exception as e:
        out["_note"] = f"Execution failed: {e}"
        return out

    ref_model = ref.SimpleModel(weights)
    want_grads = ref.run_correct_accumulation(ref_model, mb_grads, accum_steps, lr)

    if len(got_grads) == len(want_grads):
        out["accum_steps_correct"] = 1.0

    all_close = True
    if len(got_grads) == len(want_grads):
        for g_got, g_want in zip(got_grads, want_grads):
            for k in g_want:
                if k not in g_got or not ref.np.allclose(g_got[k], g_want[k], atol=1e-6):
                    all_close = False
                    break

    if all_close and out["accum_steps_correct"] == 1.0:
        out["grads_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = "Gradients or accumulation steps mismatch reference"

    return out
