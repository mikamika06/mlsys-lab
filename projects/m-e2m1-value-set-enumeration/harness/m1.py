import ref
import math

def check(workdir):
    out = {"matched_decodes": 0.0, "matched_enumerates": 0.0}
    try:
        from microscale import e2m1
    except ImportError:
        return out

    configs = [
        (1, False, False),
        (1, True, False),
        (2, False, True),
        (0, True, True)
    ]

    ok_decode = 0
    total_decode = 16 * len(configs)

    for cfg in configs:
        for val in range(16):
            want = ref.decode_e2m1(val, *cfg)
            try:
                got = e2m1.decode_e2m1(val, *cfg)
            except NotImplementedError:
                return out

            if math.isnan(want):
                if math.isnan(got):
                    ok_decode += 1
                else:
                    out["_note"] = f"decode({val}, {cfg}) want NaN, got {got}"
            elif want == got:
                ok_decode += 1
            else:
                if "_note" not in out:
                    out["_note"] = f"decode({val}, {cfg}) want {want}, got {got}"

    out["matched_decodes"] = float(ok_decode) / float(total_decode)

    ok_enum = 0
    for cfg in configs:
        want_list = ref.enumerate_values(*cfg)
        try:
            got_list = e2m1.enumerate_values(*cfg)
        except NotImplementedError:
            return out

        if len(got_list) != 16:
            out["_note"] = "enumerate_values must return exactly 16 items"
            return out

        match = True
        for w, g in zip(want_list, got_list):
            if math.isnan(w) and math.isnan(g):
                continue
            if w != g:
                match = False

        if match:
            ok_enum += 1
        else:
            if "_note" not in out:
                out["_note"] = f"enumerate({cfg}) mismatch with reference"

    out["matched_enumerates"] = float(ok_enum) / float(len(configs))
    return out
