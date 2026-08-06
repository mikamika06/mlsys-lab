import ref


def check(workdir):
    from ggufmap.mapper import map_hf_to_gguf

    out = {"names_mapped": 0.0, "total_names": float(len(ref.HF_NAMES)), "invalid_handled": 0.0}
    ok = 0
    for name in ref.HF_NAMES:
        want = ref.map_hf_to_gguf(name)
        try:
            got = map_hf_to_gguf(name)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"name {name}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"name {name} raised {type(e).__name__}: {e}"
    out["names_mapped"] = float(ok)

    bad_names = ["model.unknown.tensor", "layers.0.q_proj.weight", "foo_bar"]
    bad_ok = True
    for bad in bad_names:
        try:
            map_hf_to_gguf(bad)
            bad_ok = False
            out["_note"] = f"expected ValueError for invalid name {bad}"
            break
        except ValueError:
            pass
        except Exception as e:
            bad_ok = False
            out["_note"] = f"invalid name {bad} raised {type(e).__name__} instead of ValueError"
            break

    out["invalid_handled"] = 1.0 if bad_ok else 0.0
    return out
