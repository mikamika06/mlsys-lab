import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from vadd.kernel import run_vector_add

    out = {"correct_outputs_matched": 0.0, "underlaunch_reproduced": 0.0}

    correct_ok = True
    underlaunch_ok = True

    for i, (n, block_size) in enumerate(ref.TEST_SIZES):
        x, y = ref.generate_input_pair(n, seed=100 + i)

        want_out, want_dropped = ref.run_vector_add(x, y, block_size, grid_type="correct")
        try:
            got_out, got_dropped = run_vector_add(x, y, block_size, grid_type="correct")
        except Exception as e:
            correct_ok = False
            out["_note"] = f"run_vector_add correct failed on n={n}, block_size={block_size}: {e}"
            break

        if got_dropped != want_dropped or not np.allclose(got_out, want_out, equal_nan=True):
            correct_ok = False
            out["_note"] = f"correct run mismatch on n={n}, block_size={block_size}"
            break

        want_out_u, want_dropped_u = ref.run_vector_add(x, y, block_size, grid_type="underlaunched")
        try:
            got_out_u, got_dropped_u = run_vector_add(x, y, block_size, grid_type="underlaunched")
        except Exception as e:
            underlaunch_ok = False
            out["_note"] = f"run_vector_add underlaunched failed on n={n}, block_size={block_size}: {e}"
            break

        if got_dropped_u != want_dropped_u or not np.allclose(got_out_u, want_out_u, equal_nan=True):
            underlaunch_ok = False
            out["_note"] = f"underlaunched run mismatch on n={n}, block_size={block_size}"
            break

    if correct_ok:
        out["correct_outputs_matched"] = 1.0
    if underlaunch_ok:
        out["underlaunch_reproduced"] = 1.0

    return out
