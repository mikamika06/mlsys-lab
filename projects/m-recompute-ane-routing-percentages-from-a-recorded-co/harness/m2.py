import ref
import tempfile
import os


def check(workdir):
    from aneplan.compiler import verify_mlmodelc
    with tempfile.TemporaryDirectory() as tmp:
        ref.setup_bundle(tmp)
        want = ref.verify_mlmodelc(tmp)
        got = verify_mlmodelc(tmp)
        match = 1.0 if got == want else 0.0
        out = {"structure_match": match}
        if match == 0.0:
            out["_note"] = f"got {got}, want {want}"
        return out
