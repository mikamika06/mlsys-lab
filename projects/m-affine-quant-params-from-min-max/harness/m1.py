import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from quantizer.params import calc_affine_params

    ranges = ref.get_test_ranges()
    out = {"params_matched": 0.0, "total": float(len(ranges))}
    matched = 0

    for i, (vmin, vmax, qmin, qmax) in enumerate(ranges):
        want_scale, want_zp = ref.calc_affine_params_ref(vmin, vmax, qmin, qmax)
        try:
            got_scale, got_zp = calc_affine_params(vmin, vmax, qmin, qmax)
        except Exception as e:
            out["_note"] = f"range {i}: raised {type(e).__name__}: {e}"
            break

        scale_ok = abs(got_scale - want_scale) < 1e-6
        zp_ok = got_zp == want_zp

        if scale_ok and zp_ok:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"range {vmin},{vmax}: got ({got_scale}, {got_zp}), want ({want_scale}, {want_zp})"

    out["params_matched"] = float(matched)
    return out
