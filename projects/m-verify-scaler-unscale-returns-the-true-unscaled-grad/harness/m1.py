import ref
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    from gradscaler.scaler import GradScaler

    scale, opt, expected = ref.generate_scenario()
    scaler = GradScaler(init_scale=scale)
    try:
        grads = scaler.unscale_(opt)
        matched = 1.0
        if not grads or len(grads) != len(expected):
            matched = 0.0
        else:
            for g_grp, e_grp in zip(grads, expected):
                for g, e in zip(g_grp, e_grp):
                    if g is None or e is None or not ref.np.allclose(g, e, atol=1e-5):
                        matched = 0.0
        return {"unscale_matched": matched}
    except Exception as e:  # noqa: BLE001
        return {"unscale_matched": 0.0, "_note": str(e)}
