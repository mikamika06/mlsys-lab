import sys
import os

def check(workdir):
    sys.path.insert(0, os.path.abspath(workdir))
    import ref

    try:
        from kvtransfer.config import validate_pair
    except Exception as e:
        return {
            "valid_configs_matched": 0.0,
            "invalid_configs_caught": 0.0,
            "_note": f"Import failed: {e}"
        }

    valid_ok = True
    invalid_ok = True

    for idx, (p_cfg, c_cfg) in enumerate(ref.CONFIG_PAIRS):
        want = ref.validate_pair(p_cfg, c_cfg)
        try:
            got = validate_pair(p_cfg, c_cfg)
        except Exception as e:
            return {
                "valid_configs_matched": 0.0,
                "invalid_configs_caught": 0.0,
                "_note": f"Exception on pair {idx}: {e}"
            }

        if want["valid"]:
            if got.get("valid") != True:
                valid_ok = False
        else:
            if got.get("valid") != False or len(got.get("errors", [])) == 0:
                invalid_ok = False

    return {
        "valid_configs_matched": 1.0 if valid_ok else 0.0,
        "invalid_configs_caught": 1.0 if invalid_ok else 0.0
    }
