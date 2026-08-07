import ref
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    from gradscaler.scaler import GradScaler
    from gradscaler.verify import verify_unscaled_grad

    scale, opt, expected = ref.generate_scenario()
    scaler = GradScaler(init_scale=scale)
    try:
        err = verify_unscaled_grad(scaler, opt, expected)
        return {"max_abs_err": float(err)}
    except Exception as e:  # noqa: BLE001
        return {"max_abs_err": 999.0, "_note": str(e)}
