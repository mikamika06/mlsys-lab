import os
import tempfile
import ref
from gguf import GGUFReader


def check(workdir):
    from gguf_utils.writer import write_reference_gguf

    out = {"json_matched": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.gguf")
        learner_path = os.path.join(tmp, "learner.gguf")

        ref.generate_reference_file(ref_path)
        try:
            write_reference_gguf(learner_path)
        except Exception as e:
            out["_note"] = f"write_reference_gguf failed: {e}"
            return out

        if not os.path.exists(learner_path):
            return out

        r_ref = GGUFReader(ref_path)
        r_learn = GGUFReader(learner_path)

        ref_fields = {k: v.type for k, v in r_ref.fields.items()}
        learn_fields = {k: v.type for k, v in r_learn.fields.items()}

        ref_tensors = {t.name: (t.tensor_type, t.shape) for t in r_ref.tensors}
        learn_tensors = {t.name: (t.tensor_type, t.shape) for t in r_learn.tensors}

        if ref_fields == learn_fields and ref_tensors == learn_tensors:
            out["json_matched"] = 1.0
        else:
            out["_note"] = f"fields or tensors mismatch."
    return out
