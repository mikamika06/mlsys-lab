import sys
import ref

def check(workdir: str) -> dict:
    sys.path.insert(0, workdir)
    try:
        from ggftool.writer import GGUFWriter
    except Exception as e:
        return {"byte_exact_fraction": 0.0, "_note": f"Failed to import GGUFWriter: {e}"}

    total = len(ref.TEST_SPECS)
    matched = 0

    for spec in ref.TEST_SPECS:
        w_ref = ref.GGUFWriter(alignment=spec["alignment"])
        w_usr = GGUFWriter(alignment=spec["alignment"])

        for k, v in spec["uints"]:
            w_ref.add_uint32(k, v)
            w_usr.add_uint32(k, v)
        for k, v in spec["floats"]:
            w_ref.add_float32(k, v)
            w_usr.add_float32(k, v)
        for k, v in spec["strings"]:
            w_ref.add_string(k, v)
            w_usr.add_string(k, v)
        for name, shape, dtype_id, data in spec["tensors"]:
            w_ref.add_tensor(name, shape, dtype_id, data)
            w_usr.add_tensor(name, shape, dtype_id, data)

        bin_ref = w_ref.write()
        try:
            bin_usr = w_usr.write()
        except Exception as e:
            return {"byte_exact_fraction": 0.0, "_note": f"GGUFWriter.write raised error: {e}"}

        if bin_usr == bin_ref:
            matched += 1

    frac = matched / total
    return {"byte_exact_fraction": float(frac)}
