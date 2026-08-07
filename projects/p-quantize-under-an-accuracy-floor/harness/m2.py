import ref
import sys


def check(workdir):
    m = {"calib_effective": 0.0}
    sys_path_orig = list(sys.path)
    try:
        sys.path.insert(0, workdir)
        import quant.calib as q_calib

        x, _ = ref.generate_dataset()
        calib = q_calib.get_calibration_data(x, 20)
        if calib is not None and calib.shape[0] == 20:
            m["calib_effective"] = 1.0
    except Exception:
        pass
    finally:
        sys.path[:] = sys_path_orig
    return m
