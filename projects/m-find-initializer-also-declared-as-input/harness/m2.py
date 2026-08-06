import ref
import os


def check(workdir):
    from onnxtools.analyzer import resolve_external_ranges

    model = ref.generate_test_case_2()
    got = resolve_external_ranges(model, workdir)
    expected_path = os.path.normpath(os.path.join(workdir, "weights.bin"))
    want = {"ext_tensor": {"path": expected_path, "offset": 128, "length": 1024}}
    out = {"ranges_matched": 0.0}
    if got == want:
        out["ranges_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
