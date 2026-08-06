import numpy as np
import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from edgepipe.quant import (
        compute_quant_params,
        match_input_scale_zp,
        quantize_float_to_int8,
    )

    out = {"params_matched": 0.0, "rel_err": 1.0}
    matched = 0
    total = len(ref.TEST_CASES_QUANT)
    max_err = 0.0

    for tc in ref.TEST_CASES_QUANT:
        ref_s, ref_zp = ref.match_input_scale_zp(
            tc["mean"], tc["std"], tc["f_min"], tc["f_max"]
        )
        got_s, got_zp = match_input_scale_zp(
            tc["mean"], tc["std"], tc["f_min"], tc["f_max"]
        )

        s_err = abs(got_s - ref_s) / ref_s
        zp_err = abs(got_zp - ref_zp)
        if s_err <= 1e-4 and zp_err == 0:
            matched += 1

        dummy_data = np.linspace(
            tc["f_min"], tc["f_max"], 100, dtype=np.float32
        )
        q_ref = ref.quantize_float_to_int8(dummy_data, ref_s, ref_zp)
        q_got = quantize_float_to_int8(dummy_data, got_s, got_zp)
        err = np.max(np.abs(q_ref.astype(int) - q_got.astype(int)))
        max_err = max(max_err, float(err))

    if matched == total:
        out["params_matched"] = 1.0
    out["rel_err"] = float(max_err)
    return out
