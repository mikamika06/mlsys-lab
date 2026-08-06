import os
import tempfile
import ref


def check(workdir):
    from gguf_utils.writer import write_reference_gguf

    out = {"bytes_exact_fraction": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.gguf")
        learner_path = os.path.join(tmp, "learner.gguf")

        ref.generate_reference_file(ref_path)
        try:
            write_reference_gguf(learner_path)
        except Exception as e:
            out["_note"] = f"write_reference_gguf raised {type(e).__name__}: {e}"
            return out

        if not os.path.exists(learner_path):
            out["_note"] = "learner did not create file"
            return out

        with open(ref_path, "rb") as f:
            ref_bytes = f.read()
        with open(learner_path, "rb") as f:
            learner_bytes = f.read()

        if ref_bytes == learner_bytes:
            out["bytes_exact_fraction"] = 1.0
        else:
            out["_note"] = f"file bytes differ: ref len {len(ref_bytes)}, learner len {len(learner_bytes)}"
    return out
