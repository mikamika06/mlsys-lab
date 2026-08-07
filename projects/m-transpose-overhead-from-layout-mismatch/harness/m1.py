import numpy as np
import ref


def check(workdir):
    from edgeio.layout import measure_transpose_bytes, nchw_to_nhwc, nhwc_to_nchw

    out = {"layout_conversions_correct": 0.0, "rel_err": 1.0}

    raw = ref.sample_frames(batch_size=2, h=16, w=16, c=3, seed=123)

    try:
        nchw = nhwc_to_nchw(raw)
        nhwc_back = nchw_to_nhwc(nchw)

        want_nchw = ref.nhwc_to_nchw(raw)

        diff = np.max(np.abs(nchw.astype(np.float64) - want_nchw.astype(np.float64)))
        denom = np.max(np.abs(want_nchw.astype(np.float64))) + 1e-9
        rel_err = float(diff / denom)

        out["rel_err"] = rel_err

        roundtrip_ok = np.array_equal(raw, nhwc_back)
        contiguous_ok = nchw.flags["C_CONTIGUOUS"] and nhwc_back.flags["C_CONTIGUOUS"]

        mem_info = measure_transpose_bytes((2, 16, 16, 3), np.uint8)
        mem_ok = (mem_info.get("bytes") == 2 * 16 * 16 * 3 * 1)

        if roundtrip_ok and contiguous_ok and mem_ok and rel_err <= 1e-5:
            out["layout_conversions_correct"] = 1.0
        else:
            out["_note"] = f"Roundtrip: {roundtrip_ok}, Contiguous: {contiguous_ok}, Mem: {mem_ok}, RelErr: {rel_err}"

    except Exception as e:
        out["_note"] = f"Error during execution: {type(e).__name__}: {str(e)}"

    return out
