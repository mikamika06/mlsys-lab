import sys
import ref

def check(workdir: str) -> dict:
    sys.path.insert(0, workdir)
    try:
        from ggftool.dump import dump_json
        from ggftool.patch import patch_metadata_in_place
    except Exception as e:
        return {"dumps_matched": 0.0, "_note": f"Failed to import dump or patch module: {e}"}

    total = len(ref.TEST_SPECS)
    ok = 0

    for spec in ref.TEST_SPECS:
        w_ref = ref.GGUFWriter(alignment=spec["alignment"])
        for k, v in spec["uints"]:
            w_ref.add_uint32(k, v)
        for k, v in spec["floats"]:
            w_ref.add_float32(k, v)
        for k, v in spec["strings"]:
            w_ref.add_string(k, v)
        for name, shape, dtype_id, data in spec["tensors"]:
            w_ref.add_tensor(name, shape, dtype_id, data)

        gguf_bin = w_ref.write()

        try:
            usr_dump = dump_json(gguf_bin)
            ref_dump = ref.dump_json(gguf_bin)
        except Exception as e:
            return {"dumps_matched": 0.0, "_note": f"dump_json failed: {e}"}

        if usr_dump != ref_dump:
            return {"dumps_matched": 0.0, "_note": f"dump_json mismatch: got {usr_dump}, expected {ref_dump}"}

        patches = {spec["strings"][0][0]: "x" * len(spec["strings"][0][1])}
        try:
            patched_bin = patch_metadata_in_place(gguf_bin, patches)
            ref_patched_bin = ref.patch_metadata_in_place(gguf_bin, patches)
        except Exception as e:
            return {"dumps_matched": 0.0, "_note": f"patch_metadata_in_place failed: {e}"}

        if patched_bin != ref_patched_bin:
            return {"dumps_matched": 0.0, "_note": "patched binary does not match reference"}

        ok += 1

    return {"dumps_matched": float(ok / total)}
