import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from dcpfix.parser import parse_dcp_metadata
    from dcpfix.stitch import stitch_shards
    import numpy as np

    metadata, shards, expected = ref.generate_test_case(seed=100)
    out = {"stitching_exact": 0.0}
    try:
        parsed = parse_dcp_metadata(metadata)
        stitched = stitch_shards(shards, parsed)
        if "model.weight" in stitched:
            res = stitched["model.weight"]
            if np.allclose(res, expected):
                out["stitching_exact"] = 1.0
            else:
                out["_note"] = "Stitched tensor values do not match expected shard combination."
        else:
            out["_note"] = "Stitched dict missing expected key 'model.weight'."
    except Exception as e:
        out["_note"] = f"Stitch raised exception: {type(e).__name__}: {str(e)}"
    return out
