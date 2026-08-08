import numpy as np
import ref


def check(workdir):
    import fsdp_ckpt.converter as converter

    m = {"consolidated_match": 0.0, "shape_match": 0.0}
    ckpt = ref.get_fixture_ckpt(3)
    chunks = ref.extract_chunks(ckpt)
    metadata = ref.get_metadata()
    aligned = ref.align_shapes(chunks, metadata)

    out = converter.consolidate(aligned, metadata)
    expected = ref.consolidate(aligned, metadata)

    if out.keys() == expected.keys():
        shapes_ok = all(out[k].shape == expected[k].shape for k in out)
        if shapes_ok:
            m["shape_match"] = 1.0

        vals_ok = all(np.array_equal(out[k], expected[k]) for k in out)
        if vals_ok:
            m["consolidated_match"] = 1.0

    return m
