import ref
import numpy as np

def check(workdir):
    from zeroperf.parser import parse_log
    z2_data, z3_data = ref.LOGS
    z2_str = "\n".join([f"INFO:step_time: {t}" for t in z2_data])
    z3_str = "\n".join([f"INFO:step_time: {t}" for t in z3_data])

    parsed_z2 = parse_log(z2_str)
    parsed_z3 = parse_log(z3_str)

    ref_z2 = ref.parse_log(z2_str)
    ref_z3 = ref.parse_log(z3_str)

    ok = 0
    if len(parsed_z2) == len(ref_z2) and np.allclose(parsed_z2, ref_z2):
        ok += 1
    if len(parsed_z3) == len(ref_z3) and np.allclose(parsed_z3, ref_z3):
        ok += 1

    for scale in [0.9, 1.0, 1.1]:
        mod_data = [t * scale for t in z2_data[:20]]
        mod_str = "\n".join([f"step_time: {t}" for t in mod_data])
        if np.allclose(parse_log(mod_str), ref.parse_log(mod_str)):
            ok += 1

    out = {"logs_parsed": float(ok)}
    return out
