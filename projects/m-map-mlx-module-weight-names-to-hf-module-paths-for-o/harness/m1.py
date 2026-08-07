import ref


def check(workdir):
    from interop.mapping import map_mlx_block_weights

    out = {"mappings_matched": 0.0}
    want = ref.map_mlx_block_weights(ref.MLX_BLOCK_0, 0)
    got = map_mlx_block_weights(ref.MLX_BLOCK_0, 0)

    if isinstance(got, dict) and sorted(got.keys()) == sorted(want.keys()):
        all_match = True
        for k in want:
            if not (k in got and (got[k] == want[k]).all()):
                all_match = False
                break
        if all_match:
            out["mappings_matched"] = 1.0
        else:
            out["_note"] = "Mapped keys matched but array values differed."
    else:
        out["_note"] = f"Key mismatch. Got {sorted(got.keys()) if isinstance(got, dict) else type(got)}, expected {sorted(want.keys())}"
    return out
