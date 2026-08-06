import numpy as np
import ref


def check(workdir):
    from mxfp4.analysis import enumerate_mxfp4_grid, mxfp4_vs_q4_0_crossover

    out = {"grid_matched": 0.0, "crossover_matched": 0.0}

    grid_ok = True
    for s in [120, 127, 135]:
        want_grid = ref.ref_enumerate_mxfp4_grid(s)
        try:
            got_grid = enumerate_mxfp4_grid(s)
            if not np.allclose(want_grid, got_grid, atol=1e-6):
                grid_ok = False
                out["_note"] = f"Grid mismatch for scale {s}"
                break
        except Exception as e:
            grid_ok = False
            out["_note"] = f"enumerate_mxfp4_grid raised: {e}"
            break

    if grid_ok:
        out["grid_matched"] = 1.0

    cont_blocks = ref.generate_continuous_blocks()
    try:
        want_cross = ref.ref_mxfp4_vs_q4_0_crossover(cont_blocks)
        got_cross = mxfp4_vs_q4_0_crossover(cont_blocks)

        mse1 = abs(want_cross["mxfp4_avg_mse"] - got_cross.get("mxfp4_avg_mse", -1.0))
        mse2 = abs(want_cross["q4_0_avg_mse"] - got_cross.get("q4_0_avg_mse", -1.0))

        if mse1 < 1e-4 and mse2 < 1e-4:
            out["crossover_matched"] = 1.0
        else:
            out["_note"] = f"Crossover stats drift: want {want_cross}, got {got_cross}"
    except Exception as e:
        out["_note"] = f"mxfp4_vs_q4_0_crossover raised: {e}"

    return out
