import ref
import torch


def check(workdir):
    from optbits.measure import measure_optimizer_bytes

    model = ref.Model()
    want = ref.get_reference_measure(model)
    try:
        got = measure_optimizer_bytes(model)
    except Exception as e:
        return {
            "size_ratio_valid": 0.0,
            "bytes_counted": 0.0,
            "_note": f"raised exception: {type(e).__name__}: {str(e)[:100]}",
        }

    out = {"size_ratio_valid": 0.0, "bytes_counted": 0.0}
    if not isinstance(got, dict):
        out["_note"] = "measure_optimizer_bytes must return a dict"
        return out

    if "torch_adamw" in got and "adamw_8bit" in got:
        out["bytes_counted"] = 1.0

    ratio = got.get("size_ratio", 0.0)
    if 0.2 <= ratio <= 0.4:
        out["size_ratio_valid"] = 1.0
    else:
        out["_note"] = (
            f"size_ratio {ratio} outside expected range [0.2, 0.4], want ~{want['size_ratio']}"
        )
    return out
